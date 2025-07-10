# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
import datetime
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.pdfgenerator import PDFReportGenerator
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.interface import set_config

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
        
        self.toolkit = Toolkit(config=self.config)

        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.toolkit,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources."""
        return {
            "market": ToolNode(
                [
                    # online tools
                    self.toolkit.get_YFin_data_online,
                    self.toolkit.get_stockstats_indicators_report_online,
                    # offline tools
                    self.toolkit.get_YFin_data,
                    self.toolkit.get_stockstats_indicators_report,
                ]
            ),
            "social": ToolNode(
                [
                    # online tools
                    self.toolkit.get_stock_news_openai,
                    # offline tools
                    self.toolkit.get_reddit_stock_info,
                ]
            ),
            "news": ToolNode(
                [
                    # online tools
                    self.toolkit.get_global_news_openai,
                    self.toolkit.get_google_news,
                    # offline tools
                    self.toolkit.get_finnhub_news,
                    self.toolkit.get_reddit_news,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # online tools
                    self.toolkit.get_fundamentals_openai,
                    # offline tools
                    self.toolkit.get_finnhub_company_insider_sentiment,
                    self.toolkit.get_finnhub_company_insider_transactions,
                    self.toolkit.get_simfin_balance_sheet,
                    self.toolkit.get_simfin_cashflow,
                    self.toolkit.get_simfin_income_stmt,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)
        self.convert_markdown_reports_to_pdf(company_name, trade_date)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])
    
    # 在 trading_graph.py 中添加这个方法

    def convert_markdown_reports_to_pdf(self, company_name, trade_date):
        """在流程结束时，将所有markdown报告转换为PDF"""
        try:
            from tradingagents.agents.utils.pdfgenerator import PDFReportGenerator
            
            # 报告目录
            reports_dir = Path(f"results/{company_name}/{trade_date}/reports")
            if not reports_dir.exists():
                if self.debug:
                    print(f"⚠️ 报告目录不存在: {reports_dir}")
                return
            
            # PDF输出目录
            pdf_gen = PDFReportGenerator(output_dir=f"results/{company_name}/{trade_date}/pdf_reports")
            
            # 英文到中文的映射
            chinese_mapping = {
                "final_trade_decision.md": "最终交易决策",
                "fundamentals_report.md": "基本面分析报告",
                "investment_plan.md": "综合投资计划", 
                "market_report.md": "市场技术分析报告",
                "news_report.md": "新闻分析报告",
                "sentiment_report.md": "情绪分析报告",
                "trader_investment_plan.md": "交易员投资计划"
            }
            
            if self.debug:
                print(f"📄 开始转换markdown报告为PDF...")
                print(f"📁 源目录: {reports_dir}")
                print(f"📁 输出目录: results/{company_name}/{trade_date}/pdf_reports")
            
            all_reports = {}
            success_count = 0
            
            # 转换每个markdown文件
            for md_file in reports_dir.glob("*.md"):
                if self.debug:
                    print(f"  📝 处理: {md_file.name}")
                
                try:
                    # 读取内容
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if not content.strip():
                        if self.debug:
                            print(f"    ⚠️ 文件为空，跳过")
                        continue
                    
                    # 获取中文名称
                    chinese_name = chinese_mapping.get(md_file.name, md_file.stem)
                    
                    # 生成PDF
                    filename = f"{company_name}_{trade_date}_{chinese_name}"
                    pdf_path = pdf_gen.generate_pdf(content, filename, f"{company_name} {chinese_name}")
                    
                    if pdf_path:
                        success_count += 1
                        all_reports[chinese_name] = content
                        if self.debug:
                            print(f"    ✅ 已生成: {pdf_path.name}")
                    else:
                        if self.debug:
                            print(f"    ❌ 生成失败")
                    
                except Exception as e:
                    if self.debug:
                        print(f"    ❌ 处理失败: {e}")
            
            # 生成综合报告
            if all_reports:
                if self.debug:
                    print(f"  📋 生成综合报告...")
                
                combined_content = self._create_combined_report_simple(company_name, trade_date, all_reports)
                combined_filename = f"{company_name}_{trade_date}_综合交易分析报告"
                combined_pdf = pdf_gen.generate_pdf(combined_content, combined_filename, f"{company_name} 综合交易分析报告")
                
                if combined_pdf:
                    success_count += 1
                    if self.debug:
                        print(f"    ✅ 综合报告: {combined_pdf.name}")
            
            if self.debug:
                print(f"🎉 PDF转换完成！共生成 {success_count} 个PDF文件")
                
        except ImportError as e:
            if self.debug:
                print(f"❌ PDF生成器导入失败: {str(e)}")
        except Exception as e:
            if self.debug:
                print(f"❌ PDF转换失败: {str(e)}")

    def _create_combined_report_simple(self, company_name, trade_date, reports):
        """创建简单的综合报告"""
        import datetime
        
        combined_content = f"""# {company_name} 综合交易分析报告

    **分析日期**: {trade_date}
    **股票代码**: {company_name}
    **报告生成时间**: {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

    ---

    ## 执行摘要

    本报告基于多智能体协作分析框架，从基本面、技术面、情绪面、新闻面和风险管理等多个维度对{company_name}进行全面分析。

    ---
    """
        
        # 按逻辑顺序排列报告
        report_order = [
            "基本面分析报告",
            "市场技术分析报告",
            "新闻分析报告", 
            "情绪分析报告",
            "交易员投资计划",
            "综合投资计划",
            "最终交易决策"
        ]
        
        # 添加按顺序的报告
        for report_name in report_order:
            if report_name in reports:
                combined_content += f"\n## {report_name}\n\n{reports[report_name]}\n\n---\n"
        
        # 添加其他报告
        for report_name, content in reports.items():
            if report_name not in report_order:
                combined_content += f"\n## {report_name}\n\n{content}\n\n---\n"
        
        return combined_content
    
    def _generate_pdf_reports(self, company_name, trade_date, final_state):
        """生成PDF格式的中文报告"""
        try:
            print("🔍 开始生成PDF报告...")
            print(f"📁 输出目录: results/{company_name}/{trade_date}/pdf_reports")
            
            # 调试：查看final_state的结构
            print("🔍 final_state的键:", list(final_state.keys()))
            
            # 初始化PDF生成器
            pdf_gen = PDFReportGenerator(output_dir=f"results/{company_name}/{trade_date}/pdf_reports")
            
            # 提取各类报告内容
            reports_to_generate = {
                "基本面分析报告": self._extract_fundamentals_report(final_state),
                "市场分析报告": self._extract_market_report(final_state),  # 改为market_report
                "情绪分析报告": self._extract_sentiment_report(final_state),
                "新闻分析报告": self._extract_news_report(final_state),
                "风险管理报告": self._extract_risk_report(final_state),
                "最终交易决策": self._extract_final_decision(final_state),
                "综合投资建议": self._extract_investment_plan(final_state)
            }
            
            # 调试：显示提取到的内容
            print("📊 提取到的报告内容:")
            for name, content in reports_to_generate.items():
                if content:
                    print(f"  ✅ {name}: {len(content)} 字符")
                else:
                    print(f"  ❌ {name}: 无内容")
            
            # 生成各个PDF报告
            success_count = 0
            for report_name, content in reports_to_generate.items():
                if content:  # 只有当内容存在时才生成
                    print(f"📝 正在生成: {report_name}")
                    filename = f"{company_name}_{trade_date}_{report_name}"
                    pdf_path = pdf_gen.generate_pdf(content, filename, f"{company_name} {report_name}")
                    if pdf_path:
                        success_count += 1
                        print(f"✅ 已生成PDF报告: {pdf_path}")
                    else:
                        print(f"❌ 生成失败: {report_name}")
            
            # 生成综合报告
            if success_count > 0:
                print("📋 生成综合报告...")
                combined_content = self._create_combined_report(company_name, trade_date, reports_to_generate)
                combined_filename = f"{company_name}_{trade_date}_综合分析报告"
                combined_pdf = pdf_gen.generate_pdf(combined_content, combined_filename, f"{company_name} 综合交易分析报告")
                
                if combined_pdf:
                    print(f"✅ 已生成综合PDF报告: {combined_pdf}")
            else:
                print("⚠️ 没有内容可生成综合报告")
            
            print(f"🎉 PDF生成完成，共生成 {success_count} 个报告")
            print(f"📁 请检查目录: results/{company_name}/{trade_date}/pdf_reports/")
                    
        except Exception as e:
            print(f"❌ PDF生成失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _extract_market_report(self, final_state):
        """从状态中提取市场分析报告"""
        try:
            if "market_report" in final_state:
                print("✅ 找到market_report")
                return final_state["market_report"]
            print("❌ 未找到market_report")
            return None
        except Exception as e:
            print(f"❌ 提取market_report失败: {e}")
            return None

    def _extract_fundamentals_report(self, final_state):
        """从状态中提取基本面分析报告"""
        try:
            if "fundamentals_report" in final_state:
                print("✅ 找到fundamentals_report")
                return final_state["fundamentals_report"]
            print("❌ 未找到fundamentals_report")
            return None
        except Exception as e:
            print(f"❌ 提取fundamentals_report失败: {e}")
            return None

    def _extract_sentiment_report(self, final_state):
        """从状态中提取情绪分析报告"""
        try:
            if "sentiment_report" in final_state:
                print("✅ 找到sentiment_report")
                return final_state["sentiment_report"]
            print("❌ 未找到sentiment_report")
            return None
        except Exception as e:
            print(f"❌ 提取sentiment_report失败: {e}")
            return None

    def _extract_news_report(self, final_state):
        """从状态中提取新闻分析报告"""
        try:
            if "news_report" in final_state:
                print("✅ 找到news_report")
                return final_state["news_report"]
            print("❌ 未找到news_report")
            return None
        except Exception as e:
            print(f"❌ 提取news_report失败: {e}")
            return None

    def _extract_risk_report(self, final_state):
        """从状态中提取风险管理报告"""
        try:
            # 检查多个可能的字段
            possible_fields = ["risk_report", "risk_debate_state", "risk_management"]
            for field in possible_fields:
                if field in final_state:
                    print(f"✅ 找到{field}")
                    content = final_state[field]
                    if isinstance(content, dict):
                        # 如果是字典，转换为可读格式
                        return f"""# 风险管理报告

    ## 风险评估结果
    {json.dumps(content, ensure_ascii=False, indent=2)}
    """
                    else:
                        return str(content)
            print("❌ 未找到风险管理相关报告")
            return None
        except Exception as e:
            print(f"❌ 提取risk_report失败: {e}")
            return None

    def _extract_final_decision(self, final_state):
        """从状态中提取最终交易决策"""
        try:
            if "final_trade_decision" in final_state:
                print("✅ 找到final_trade_decision")
                decision = final_state["final_trade_decision"]
                if isinstance(decision, dict):
                    return f"""# 最终交易决策

    ## 决策结果
    **股票代码**: {self.ticker}
    **决策**: {decision.get('action', 'HOLD')}
    **建议价格**: {decision.get('price', 'N/A')}
    **风险等级**: {decision.get('risk_level', 'N/A')}

    ## 决策依据
    {decision.get('reasoning', '无详细说明')}
    """
                else:
                    return f"""# 最终交易决策

    {str(decision)}
    """
            print("❌ 未找到final_trade_decision")
            return None
        except Exception as e:
            print(f"❌ 提取final_decision失败: {e}")
            return None

    def _extract_investment_plan(self, final_state):
        """从状态中提取投资计划"""
        try:
            # 检查多个可能的字段
            possible_fields = ["investment_plan", "trader_investment_plan", "trader_investment_decision"]
            for field in possible_fields:
                if field in final_state:
                    print(f"✅ 找到{field}")
                    content = final_state[field]
                    if isinstance(content, dict):
                        return f"""# 投资计划

    ## 投资建议
    {json.dumps(content, ensure_ascii=False, indent=2)}
    """
                    else:
                        return str(content)
            print("❌ 未找到投资计划相关内容")
            return None
        except Exception as e:
            print(f"❌ 提取investment_plan失败: {e}")
            return None

    def _create_combined_report(self, company_name, trade_date, reports):
        """创建综合报告"""
        combined_content = f"""# {company_name} 综合交易分析报告

    **分析日期**: {trade_date}
    **股票代码**: {company_name}
    **报告生成时间**: {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

    ---

    ## 执行摘要

    本报告基于多智能体协作分析框架，从基本面、技术面、情绪面、新闻面和风险管理等多个维度对{company_name}进行全面分析。

    ---
    """
        
        # 添加各个报告部分
        for report_name, content in reports.items():
            if content:
                combined_content += f"\n## {report_name}\n\n{content}\n\n---\n"
        
        return combined_content

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
