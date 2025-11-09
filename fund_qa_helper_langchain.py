#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行代码前需要先：(本代码使用的langchain框架做的)
pip install langchain
pip install langchain-community
pip install dashscope
"""

"""
私募基金运作指引问答助手 - 反应式智能体实现

适合反应式架构的私募基金问答助手，使用Agent模式实现主动思考和工具选择。
"""

import re
from typing import List, Dict, Any, Union
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain_community.llms import Tongyi
from langchain import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
import os

# 设置通义千问API密钥
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# 简化的私募基金规则数据库
FUND_RULES_DB = [
    {
        "id": "rule001",
        "category": "设立与募集",
        "question": "私募基金的合格投资者标准是什么？",
        "answer": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于100万元且符合下列条件之一的单位和个人：\n1. 净资产不低于1000万元的单位\n2. 金融资产不低于300万元或者最近三年个人年均收入不低于50万元的个人"
    },
    {
        "id": "rule002",
        "category": "设立与募集",
        "question": "私募基金的最低募集规模要求是多少？",
        "answer": "私募证券投资基金的最低募集规模不得低于人民币1000万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。"
    },
    {
        "id": "rule014",
        "category": "监管规定",
        "question": "私募基金管理人的风险准备金要求是什么？",
        "answer": "私募证券基金管理人应当按照管理费收入的10%计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。"
    }
]

# 定义上下文QA模板
CONTEXT_QA_TMPL = """
你是私募基金问答助手。请根据以下信息回答问题：

信息：{context}
问题：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

# 定义超出知识库范围问题的回答模板
OUTSIDE_KNOWLEDGE_TMPL = """
你是私募基金问答助手。用户的问题是关于私募基金的，但我们的知识库中没有直接相关的信息。
请首先明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"，
然后，如果你有相关知识，可以以"根据我的经验"或"一般来说"等方式提供一些通用信息，
并建议用户查阅官方资料或咨询专业人士获取准确信息。

用户问题：{query}
缺失的知识主题：{missing_topic}
"""
OUTSIDE_KNOWLEDGE_PROMPT = PromptTemplate(
    input_variables=["query", "missing_topic"],
    template=OUTSIDE_KNOWLEDGE_TMPL,
)

# 私募基金问答数据源,跟9-LANGCHAIN:多任务应用开发 里的5-product_llm.py相关代码一样
class FundRulesDataSource:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.rules_db = FUND_RULES_DB

    # 工具1：通过关键词搜索相关规则
    def search_rules_by_keywords(self, keywords: str) -> str:
        """通过关键词搜索相关私募基金规则"""
        keywords = keywords.strip().lower() #去除前后空白字符并转为小写
        keyword_list = re.split(r'[,，\s]+', keywords)  #\s：任何空白字符（包括空格、制表符、换行符等）
        
        matched_rules = []
        for rule in self.rules_db:
            rule_text = (rule["category"] + " " + rule["question"]).lower()
            match_count = sum(1 for kw in keyword_list if kw in rule_text)
            if match_count > 0:
                matched_rules.append((rule, match_count))  
                #每个元素是一个元组，每个规则匹配到的关键词数量，eg.({"category": "基金", "question": "如何申购"}, 3)
        
        matched_rules.sort(key=lambda x: x[1], reverse=True)
        #根据元组的第二个元素（即 match_count）降序排序，优先显示匹配次数多的规则。
        
        if not matched_rules:
            return "未找到与关键词相关的规则。"
        
        result = []
        for rule, _ in matched_rules[:2]:  #只显示前2条匹配的规则
            result.append(f"类别: {rule['category']}\n问题: {rule['question']}\n答案: {rule['answer']}")
        
        return "\n\n".join(result)

    # 工具2：根据规则类别查询
    def search_rules_by_category(self, category: str) -> str:
        """根据规则类别查询私募基金规则"""
        category = category.strip()  #去除前后空白字符
        matched_rules = []
        
        for rule in self.rules_db:
            if category.lower() in rule["category"].lower():
                matched_rules.append(rule)
        
        if not matched_rules:
            return f"未找到类别为 '{category}' 的规则。"
        
        result = []
        for rule in matched_rules:
            result.append(f"问题: {rule['question']}\n答案: {rule['answer']}")
        
        return "\n\n".join(result)

    # 工具3：直接回答用户问题
    def answer_question(self, query: str) -> str:
        """直接回答用户关于私募基金的问题"""
        query = query.strip()
        
        best_rule = None
        best_score = 0
        
        for rule in self.rules_db:
            query_words = set(query.lower().split())  #如 "how to invest" → ["how", "to", "invest"]
            rule_words = set((rule["question"] + " " + rule["category"]).lower().split())
            #规则文本分词：将规则的 question 和 category 字段拼接后，同样分词并转为集合（如 "基金 如何申购" → {"基金", "如何", "申购"}）
            
            common_words = query_words.intersection(rule_words)
            """我加的注释：通过集合的 intersection 方法，找出 用户查询和规则文本共有的词汇。
            例如：
                用户查询：{"私募", "基金", "门槛"}
                规则文本：{"私募", "基金", "申购"}
                共同词汇：{"私募", "基金"}
            """

            
            score = len(common_words) / max(1, len(query_words))  #感觉分母应该用rule_words？？
            if score > best_score:
                best_score = score
                best_rule = rule
        
        if best_score < 0.2 or best_rule is None: #如果没有找到匹配的规则，或者匹配度太低，则返回超出知识库范围的回答
            # 识别问题主题
            missing_topic = self._identify_missing_topic(query)
            prompt = OUTSIDE_KNOWLEDGE_PROMPT.format(
                query=query,
                missing_topic=missing_topic
            )
            # 直接通过LLM获取回答可能导致输出格式与Agent期望不符
            # 将回答包装为AgentFinish格式而不是返回给Agent处理
            #response = self.llm(prompt)
            response = self.llm.invoke(prompt)
            # 返回格式化后的回答，让Agent直接返回最终结果
            return f"这个问题超出了知识库范围。\n\n{response}"
        
        context = best_rule["answer"]
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        
        #return self.llm(prompt)
        response = self.llm.invoke(prompt)
    
    def _identify_missing_topic(self, query: str) -> str:
        """识别查询中缺失的知识主题"""
        #我加的：即，如果用户提问的是这些关键词，则是超出知识库范围的问题（老师自定义死的，呵呵）
        # 简单的主题提取逻辑
        query = query.lower()
        if "投资" in query and "资产" in query:
            return "私募基金可投资的资产类别"
        elif "公募" in query and "区别" in query:
            return "私募基金与公募基金的区别"
        elif "退出" in query and ("机制" in query or "方式" in query):
            return "创业投资基金的退出机制"
        elif "费用" in query and "结构" in query:
            return "私募基金的费用结构"
        elif "托管" in query:
            return "私募基金资产托管"
        # 如果无法确定具体主题，使用通用表述
        return "您所询问的具体主题"


# 定义Agent模板
AGENT_TMPL = """你是一个私募基金问答助手，请根据用户的问题选择合适的工具来回答。

你可以使用以下工具：

{tools}

按照以下格式回答问题：

---
Question: 用户的问题
Thought: 我需要思考如何回答这个问题
Action: 工具名称
Action Input: 工具的输入
Observation: 工具返回的结果
...（这个思考/行动/行动输入/观察可以重复几次）
Thought: 现在我知道答案了
Final Answer: 给用户的最终答案
---

注意：
1. 如果知识库中没有相关信息，请明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"
2. 如果你基于自己的知识提供补充信息，请用"根据我的经验"或"一般来说"等前缀明确标识
3. 回答要专业、简洁、准确

Question: {input}
{agent_scratchpad}
"""
#agent_scratchpad是草稿的意思

# 自定义Prompt模板，跟9-LANGCHAIN:多任务应用开发 里的5-product_llm.py相关代码一样
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        return self.template.format(**kwargs)


# 自定义输出解析器
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        #print(f"LLM输出: {llm_output}")
        
        # 如果输出包含"Final Answer:"，直接处理为最终答案
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
            
        # 如果输出以"对不起"或"抱歉"开头，可能是LLM直接给出了回答而非遵循格式
        if llm_output.strip().startswith("对不起") or llm_output.strip().startswith("抱歉"):
            # 直接将其视为最终答案
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=f"Direct response detected: {llm_output}"
            )
            
        # 如果输出包含明确的知识边界声明，也视为最终答案
        knowledge_boundary_phrases = [
            "在我的知识库中没有",
            "超出了我的知识范围",
            "我没有相关信息",
            "根据我的经验"
        ]
        
        for phrase in knowledge_boundary_phrases:
            if phrase in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.strip()},
                    log=f"Knowledge boundary response detected: {llm_output}"
                )

        # 尝试解析Action和Action Input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # 如果不符合格式但内容较长，可能是LLM直接给出了详细回答
            if len(llm_output.strip()) > 50:  # 假设超过50个字符的回答为实质性内容
                return AgentFinish(
                    return_values={"output": llm_output.strip()},
                    log=f"Long unstructured response detected: {llm_output}"
                )
            # 真正无法解析的情况
            raise ValueError(f"无法解析LLM输出: `{llm_output}`")
        
        action = match.group(1).strip()
        action_input = match.group(2)
        
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def create_fund_qa_agent():
    # 定义LLM
    llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=DASHSCOPE_API_KEY)
    #from langchain.chat_models import init_chat_model
    #llm = init_chat_model("Qwen-Turbo-2025-04-28", model_provider="Tongyi") #不行，init_chat_model方法不支持Tongyi模型
    
    # 创建数据源
    fund_rules_source = FundRulesDataSource(llm)
    
    # 定义工具，里面的每一个Tool是上面import的一个Tool类
    tools = [
        Tool(
            name="关键词搜索",
            func=fund_rules_source.search_rules_by_keywords,
            description="当需要通过关键词搜索私募基金规则时使用，输入应为相关关键词",
        ),
        Tool(
            name="类别查询",
            func=fund_rules_source.search_rules_by_category,
            description="当需要查询特定类别的私募基金规则时使用，输入应为类别名称。类别名称有两种：设立与募集, 监管规定",
        ),
        Tool(
            name="回答问题",
            func=fund_rules_source.answer_question,
            description="当能够直接回答用户问题时使用，输入应为完整的用户问题",
        ),
    ]
    
    # 创建Agent提示模板
    agent_prompt = CustomPromptTemplate(
        template=AGENT_TMPL,
        tools=tools,
        input_variables=["input", "intermediate_steps"],
    )
    
    # 创建输出解析器
    output_parser = CustomOutputParser()
    
    # 创建LLM链
    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
    #运行的时候，这里会提示：LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and 
    # will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.

    #llm_chain = agent_prompt | llm   #直接这样不行，后面再研究，可能需要agent_prompt和llm都实现Runnable接口
    
    #from langchain_core.runnables import RunnableSequence
    #llm_chain = RunnableSequence(first=llm, last=agent_prompt)  #不行，跟下面的LLMSingleActionAgent不兼容
    
    # 获取工具名称
    tool_names = [tool.name for tool in tools]
    
    # 创建Agent
    agent = LLMSingleActionAgent(  #运行的时候，这里会提示你用LangGraph
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )
    
    # 创建Agent执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )
    
    return agent_executor


if __name__ == "__main__":
    # 创建Agent
    fund_qa_agent = create_fund_qa_agent()
    
    print("=== 私募基金运作指引问答助手（反应式智能体）===\n")
    print("使用模型：Qwen-Turbo-2025-04-28\n")
    print("您可以提问关于私募基金的各类问题，输入'退出'结束对话\n")
    
    # 主循环
    while True:
        try:
            user_input = input("请输入您的问题：")
            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("感谢使用，再见！")
                break
            
            response = fund_qa_agent.run(user_input)
            #response = fund_qa_agent.invoke(user_input) #这个方法最终回答的结果是个json,\n没有转义，不直观
            print(f"回答: {response}\n")
            print("-" * 40)
        except KeyboardInterrupt:
            print("\n程序已中断，感谢使用！")
            break
        except Exception as e:
            print(f"发生错误：{e}")
            print("请尝试重新提问或更换提问方式。") 
