"""A1pro - 扩展的 CAi Agent，自动加载 additional_tools 和 skills"""

import importlib
import inspect

from base_CAi.agent.a1 import A1
from CAi.CAi_agent.skills import SkillLoader
from CAi.logger import get_logger


class A1pro(A1):
    """
    A1pro - cai agent

    特性：
    - 自动加载 additional_tools 模块中的所有工具
    - 支持复杂依赖的工具
    - 支持手动排除某些工具
    - 支持延迟加载（lazy loading）
    """

    def __init__(
        self,
        *args,
        auto_load_tools: bool = True,
        tools_module: str = "CAi.additional_tools",
        exclude_tools: list[str] = None,
        lazy_load: bool = False,
        auto_load_skills: bool = True,
        skills_dir: str | None = None,
        exclude_skills: list[str] = None,
        **kwargs,
    ):
        """
        初始化 A1pro Agent

        Args:
            *args: 传递给父类 A1 的位置参数
            auto_load_tools: 是否自动加载工具（默认 True）
            tools_module: 工具模块路径（默认 "CAi.additional_tools"）
            exclude_tools: 要排除的工具名称列表
            lazy_load: 是否延迟加载工具（首次使用时才加载）
            auto_load_skills: 是否自动加载技能（默认 True）
            skills_dir: 技能文件目录（默认使用 CAi/skills）
            exclude_skills: 要排除的技能 ID 列表
            **kwargs: 传递给父类 A1 的关键字参数
        """
        # ⚠️ 重要：在调用 super().__init__() 之前初始化这些属性
        # 因为父类的 __init__ 会调用 configure()，而子类重写的 configure() 需要这些属性
        self._name = "A1pro"
        self.tools_module = tools_module
        self.exclude_tools = exclude_tools or []
        self.lazy_load = lazy_load

        # 存储已加载的工具
        self._loaded_tools = {}
        self._tools_metadata = {}

        # 初始化 SkillLoader（必须在 super().__init__() 之前）
        self.skill_loader = SkillLoader(skills_dir)
        self.exclude_skills = exclude_skills or []
        self._auto_load_skills = auto_load_skills  # 保存标志，稍后使用

        # Initialize logger
        self.logger = get_logger("CAi.A1pro")

        # 现在可以安全地调用父类初始化
        super().__init__(*args, **kwargs)

        # 自动加载工具
        if auto_load_tools:
            if lazy_load:
                self.logger.info("📦 A1pro 初始化完成（延迟加载模式）")
                self._discover_tools()  # 只发现，不加载
            else:
                self.logger.info(f"📦 A1pro 正在加载 {tools_module} 中的工具...")
                self._load_all_tools()

        # 自动加载技能（只加载元数据）
        if auto_load_skills:
            self._load_skill_summaries()

        # 统一更新系统提示词（如果加载了工具或技能）
        if auto_load_tools or auto_load_skills:
            self.logger.info("🔄 更新系统提示词...")
            self.configure()

    def _discover_tools(self):
        """发现但不加载工具（用于延迟加载）"""
        try:
            # 导入工具模块
            module = importlib.import_module(self.tools_module)

            # 获取模块中的所有函数
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # 跳过私有函数和排除的工具
                if name.startswith("_") or name in self.exclude_tools:
                    continue

                # 存储工具元数据
                self._tools_metadata[name] = {"function": obj, "module": module, "loaded": False}

            self.logger.info(f"🔍 发现 {len(self._tools_metadata)} 个工具（延迟加载）")

        except Exception as e:
            self.logger.error(f"⚠️  发现工具时出错: {e}")
            self.logger.exception("Tool discovery failed")

    def _load_all_tools(self):
        """加载所有工具"""
        try:
            # 导入工具模块（不使用 reload，避免副作用）
            module = importlib.import_module(self.tools_module)

            # 获取模块中的所有函数
            tools_found = 0
            tools_loaded = 0

            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # 跳过私有函数和排除的工具
                if name.startswith("_") or name in self.exclude_tools:
                    continue

                tools_found += 1

                try:
                    # 添加工具到 agent
                    self.add_tool(obj)
                    self._loaded_tools[name] = obj
                    tools_loaded += 1
                    self.logger.debug(f"  ✓ 已加载: {name}")

                except Exception as e:
                    self.logger.warning(f"  ✗ 加载失败: {name} - {e}")

            self.logger.info(f"✅ 成功加载 {tools_loaded}/{tools_found} 个工具")

            if tools_loaded > 0:
                self.logger.info("可用工具列表:")
                for tool_name in self._loaded_tools.keys():
                    self.logger.info(f"  - {tool_name}")

        except ModuleNotFoundError as e:
            self.logger.error(f"⚠️  未找到工具模块 '{self.tools_module}': {e}")
            self.logger.info(f"提示: 请确保 {self.tools_module.replace('.', '/')} 目录存在")
        except Exception as e:
            self.logger.error(f"⚠️  加载工具时出错: {e}")
            self.logger.exception("Tool loading failed")

    def _load_skill_summaries(self):
        """加载技能摘要（只加载 name, description, metadata）"""
        try:
            self.logger.info("🎯 A1pro 正在加载技能摘要...")

            # 获取所有技能的摘要
            summaries = self.skill_loader.get_skill_summaries()

            # 过滤排除的技能
            filtered_summaries = [s for s in summaries if s["id"] not in self.exclude_skills]

            skills_loaded = len(filtered_summaries)

            if skills_loaded > 0:
                self.logger.info(f"✅ 成功加载 {skills_loaded} 个技能摘要")
                self.logger.info("可用技能列表:")
                for summary in filtered_summaries:
                    self.logger.info(f"  - {summary['name']}")

                # 将 get_skill_content 作为工具添加到 agent
                self._register_skill_tool()
            else:
                self.logger.warning("⚠️  未找到可用的技能文件")

        except Exception as e:
            self.logger.error(f"⚠️  加载技能时出错: {e}")
            self.logger.exception("Skill loading failed")

    def _get_skill_docs_for_prompt(self) -> list[dict]:
        """把 skill 摘要转换为 know_how_docs 格式，供 _generate_system_prompt 使用。"""
        if not hasattr(self, "skill_loader") or not self.skill_loader:
            return []
        summaries = self.skill_loader.get_skill_summaries()
        if hasattr(self, "exclude_skills"):
            summaries = [s for s in summaries if s["id"] not in self.exclude_skills]

        docs = []
        for s in summaries:
            lines = [f"**ID**: `{s['id']}`", f"**Description**: {s['description']}"]
            meta = s.get("metadata", {})
            if meta.get("required_tools"):
                lines.append(f"**Required Tools**: {meta['required_tools']}")
            if meta.get("category"):
                lines.append(f"**Category**: {meta['category']}")
            lines.append(f"**To get detailed workflow**: `get_skill_content('{s['id']}')`")
            docs.append({"name": f"Skill: {s['name']}", "content": "\n".join(lines)})
        return docs

    def _build_system_prompt_context(self, selected_resources=None) -> dict:
        """Override to inject skills via the standard know_how_docs pipeline."""
        context = super()._build_system_prompt_context(selected_resources)
        skill_docs = self._get_skill_docs_for_prompt()
        if skill_docs:
            existing = context.get("know_how_docs") or []
            context["know_how_docs"] = existing + skill_docs
            self.logger.info(f"✅ 已将 {len(skill_docs)} 个技能摘要注入系统提示词")
        return context

    def _register_skill_tool(self):
        """Skills are loaded via template_tools.py (get_skill_content / list_available_skills).
        Nothing extra to register here — kept as a hook for subclasses."""
        pass

    def get_skill_content(self, skill_id: str) -> dict | None:
        """
        获取技能的完整内容（延迟加载）- 供外部调用

        Args:
            skill_id: 技能 ID

        Returns:
            技能的完整信息，包括 content
        """
        return self.skill_loader.get_skill_by_id(skill_id)

    def reload_tools(self):
        """重新加载所有工具（用于开发时热更新）"""
        self.logger.info(f"🔄 重新加载 {self.tools_module} 中的工具...")

        # 清除已加载的工具
        for tool_name in list(self._loaded_tools.keys()):
            try:
                self.remove_custom_tool(tool_name)
            except Exception:
                pass

        self._loaded_tools.clear()

        # 重新加载
        self._load_all_tools()
        # ⭐ 重新配置以更新系统提示词
        self.logger.info("🔄 更新系统提示词...")
        self.configure()

    def reload_skills(self):
        """重新加载所有技能（用于开发时热更新）"""
        self.logger.info("🔄 重新加载技能...")
        self.skill_loader.reload()
        self._load_skill_summaries()
        self.logger.info("🔄 更新系统提示词...")
        self.configure()

    def add_tool_from_file(self, filepath: str, function_name: str = None):
        """
        从文件中加载特定工具

        Args:
            filepath: Python 文件路径
            function_name: 要加载的函数名（如果为 None，加载所有函数）
        """
        try:
            # 读取文件
            with open(filepath, encoding="utf-8") as f:
                code = f.read()

            # 创建临时模块
            import types

            temp_module = types.ModuleType("temp_tools")

            # 执行代码
            exec(code, temp_module.__dict__)

            # 加载函数
            if function_name:
                if hasattr(temp_module, function_name):
                    func = getattr(temp_module, function_name)
                    self.add_tool(func)
                    self.logger.info(f"✓ 从 {filepath} 加载工具: {function_name}")
                else:
                    self.logger.warning(f"✗ 在 {filepath} 中未找到函数: {function_name}")
            else:
                # 加载所有函数
                for name, obj in inspect.getmembers(temp_module, inspect.isfunction):
                    if not name.startswith("_"):
                        self.add_tool(obj)
                        self.logger.info(f"✓ 从 {filepath} 加载工具: {name}")

        except Exception as e:
            self.logger.error(f"✗ 从文件加载工具失败: {e}")
            self.logger.exception("Tool loading from file failed")

    def list_available_tools(self):
        """列出所有可用的工具"""
        print("\n" + "=" * 60)
        print("📋 A1pro 可用工具列表")
        print("=" * 60)

        # CAi 内置工具
        print("\n🔧 CAi 内置工具:")
        if hasattr(self, "module2api"):
            for module_name, tools in self.module2api.items():
                if module_name not in ["custom_tools", "mcp_servers"]:
                    print(f"\n  [{module_name}]")
                    for tool in tools[:3]:  # 只显示前 3 个
                        if isinstance(tool, dict) and "name" in tool:
                            print(f"    - {tool['name']}")
                    if len(tools) > 3:
                        print(f"    ... 还有 {len(tools) - 3} 个工具")

        # 自定义工具
        if self._loaded_tools:
            print(f"\n🎯 Additional Tools ({len(self._loaded_tools)} 个):")
            for tool_name, tool_func in self._loaded_tools.items():
                doc = tool_func.__doc__ or "无描述"
                # 获取第一行文档
                first_line = doc.strip().split("\n")[0]
                print(f"  - {tool_name}: {first_line}")

        # MCP 工具
        if hasattr(self, "_custom_tools") and self._custom_tools:
            mcp_tools = [name for name, info in self._custom_tools.items() if "mcp_servers" in info.get("module", "")]
            if mcp_tools:
                print(f"\n🔌 MCP 工具 ({len(mcp_tools)} 个):")
                for tool_name in mcp_tools:
                    print(f"  - {tool_name}")

        print("\n" + "=" * 60)

    def list_available_skills(self):
        """列出所有可用的技能"""
        summaries = self.skill_loader.get_skill_summaries()

        # 过滤排除的技能
        summaries = [s for s in summaries if s["id"] not in self.exclude_skills]

        if not summaries:
            print("📋 暂无可用技能")
            return

        print("\n" + "=" * 60)
        print(f"🎯 A1pro 可用技能列表 ({len(summaries)} 个)")
        print("=" * 60)

        for summary in summaries:
            print(f"\n  🔹 {summary['name']}")
            print(f"     {summary['description']}")

            # 显示元数据
            metadata = summary.get("metadata", {})
            if "required_tools" in metadata:
                print(f"     需要工具: {metadata['required_tools']}")
            if "category" in metadata:
                print(f"     分类: {metadata['category']}")

        print("\n" + "=" * 60)

    def get_skill_info(self, skill_id: str):
        """获取技能的详细信息（会加载完整内容）"""
        skill = self.skill_loader.get_skill_by_id(skill_id)

        if not skill:
            self.logger.warning(f"⚠️  未找到技能: {skill_id}")
            self.logger.info("提示: 使用 list_available_skills() 查看所有可用技能")
            return

        # 使用 loader 的打印方法
        self.skill_loader.print_skill_info(skill_id)

    def configure(self, self_critic=False, test_time_scale_round=0):
        """Override configure to ensure skill_loader is ready before building the prompt."""
        if not hasattr(self, "skill_loader") or self.skill_loader is None:
            super().configure(self_critic=self_critic, test_time_scale_round=test_time_scale_round)
            return
        # _build_system_prompt_context is already overridden above to inject skills
        super().configure(self_critic=self_critic, test_time_scale_round=test_time_scale_round)

    def get_tool_info(self, tool_name: str):
        """获取工具的详细信息"""
        # 检查是否在已加载的工具中
        if tool_name in self._loaded_tools:
            func = self._loaded_tools[tool_name]
            print(f"\n📖 工具信息: {tool_name}")
            print("=" * 60)
            print(f"模块: {self.tools_module}")
            print(f"文档:\n{func.__doc__ or '无文档'}")
            print(f"签名: {inspect.signature(func)}")
            print("=" * 60)
        else:
            print(f"⚠️  未找到工具: {tool_name}")
            print("提示: 使用 list_available_tools() 查看所有可用工具")

    def launch_new_gradio_demo(self, thread_id=42, share=False, server_name="0.0.0.0", require_verification=False):
        """
        启动 Gradio UI (视图层已完全解耦)
        """
        try:
            import gradio as gr
        except ImportError as e:
            raise ImportError("Gradio is not installed. Please install it with: pip install gradio") from e

        from .ui import AgentGradioUI

        ui_controller = AgentGradioUI(agent=self, thread_id=thread_id, require_verification=require_verification)

        # 获取构建好的 gr.Blocks 对象
        demo = ui_controller.build_ui()

        # 启动服务
        self.logger.info(f"🚀 Launching Gradio demo on {server_name}:7860")
        demo.launch(share=share, server_name=server_name)

    def launch_web_ui(self, backend_port=7000):
        """
        启动现代化 Web UI (React + FastAPI)

        Args:
            backend_port: 后端 API 端口（默认 8000）

        Example:
            >>> agent = A1pro()
            >>> agent.launch_web_ui()
        """
        from CAi.bio_agent.launch_web_ui import launch_web_ui

        launch_web_ui(self, backend_port)
