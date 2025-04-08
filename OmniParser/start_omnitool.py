import os
import sys
import time
import subprocess
import requests
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OmniToolStarter:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.omniparser_server_url = "http://127.0.0.1:8000"
        self.omnibox_url = "http://127.0.0.1:5000"
        self.gradio_url = "http://127.0.0.1:7888"

    def check_conda_env(self):
        """检查conda环境是否正确设置"""
        try:
            # 检查是否在conda环境中运行
            if 'conda' in sys.version or 'Continuum' in sys.version:
                # 判断环境名称是否为'omni'
                env_name = os.environ.get('CONDA_DEFAULT_ENV', '')
                if env_name == 'omni':
                    logger.info("已在'omni' conda环境中运行")
                    return True
                else:
                    logger.warning(f"当前conda环境为'{env_name}'，而非'omni'环境")
            
            # 检查是否已安装必要的依赖
            try:
                import torch
                import gradio
                import fastapi
                logger.info("已安装必要的依赖")
                return True
            except ImportError as e:
                logger.error(f"缺少必要的依赖: {e}")
                logger.error("请确保已安装所有依赖: pip install -r requirements.txt")
                return False
        except Exception as e:
            logger.error(f"检查环境时出错: {e}")
            return False

    def start_omniparser_server(self):
        """启动OmniParser服务器"""
        try:
            server_path = self.base_path / "omnitool" / "omniparserserver"
            logger.info("正在启动OmniParser服务器...")
            
            # 在Windows上使用
            if sys.platform == 'win32':
                subprocess.Popen(
                    ["python", "-m", "omniparserserver"],
                    cwd=server_path,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # 在Linux/Mac上使用
                subprocess.Popen(
                    ["python", "-m", "omniparserserver"],
                    cwd=server_path
                )
            return True
        except Exception as e:
            logger.error(f"启动OmniParser服务器时出错: {e}")
            return False

    def start_omnibox(self):
        """启动OmniBox"""
        try:
            scripts_path = self.base_path / "omnitool" / "omnibox" / "scripts"
            logger.info("正在启动OmniBox...")
            
            # 检查manage_vm.sh文件是否存在
            manage_vm_path = scripts_path / "manage_vm.sh"
            if not manage_vm_path.exists():
                logger.error(f"找不到manage_vm.sh文件: {manage_vm_path}")
                return False
                
            # 检查Git Bash是否安装
            git_bash_path = "C:\\Program Files\\Git\\bin\\bash.exe"
            if not os.path.exists(git_bash_path):
                logger.error("找不到Git Bash，请先安装Git for Windows")
                logger.error("下载地址: https://git-scm.com/download/win")
                return False
            
            # 使用Git Bash执行manage_vm.sh
            logger.info("使用Git Bash执行manage_vm.sh")
            
            # 将Windows路径转换为Git Bash可用的路径格式
            bash_scripts_path = str(scripts_path).replace('\\', '/').replace('C:', '/c')
            bash_manage_vm_path = f"{bash_scripts_path}/manage_vm.sh"
            
            # 确保脚本有执行权限
            try:
                # 尝试使用chmod命令设置执行权限
                chmod_cmd = [git_bash_path, "-c", f"chmod +x {bash_manage_vm_path}"]
                subprocess.run(chmod_cmd, check=True)
                logger.info("已设置manage_vm.sh的执行权限")
            except subprocess.CalledProcessError:
                logger.warning("无法设置manage_vm.sh的执行权限，尝试直接执行")
            
            # 构建完整的命令
            cmd = [git_bash_path, "-c", f"cd {bash_scripts_path} && bash {bash_manage_vm_path} start"]
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 使用subprocess.Popen启动进程，并捕获输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            # 等待一段时间，确保脚本开始执行
            time.sleep(5)
            
            # 获取输出信息
            stdout, stderr = process.communicate()
            
            # 检查输出中是否包含成功信息
            if "VM started" in stdout:
                logger.info("manage_vm.sh脚本执行成功")
                return True
            else:
                logger.error(f"manage_vm.sh脚本执行失败")
                if stdout:
                    logger.error(f"标准输出: {stdout}")
                if stderr:
                    logger.error(f"错误输出: {stderr}")
                return False
            
        except Exception as e:
            logger.error(f"启动OmniBox时出错: {e}")
            return False

    def start_gradio(self):
        """启动Gradio界面"""
        try:
            gradio_path = self.base_path / "omnitool" / "gradio"
            logger.info("正在启动Gradio界面...")
            
            # 在Windows上使用
            if sys.platform == 'win32':
                subprocess.Popen(
                    ["python", "app.py", "--windows_host_url", "localhost:8006", "--omniparser_server_url", "localhost:8000"],
                    cwd=gradio_path,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                # 在Linux/Mac上使用
                subprocess.Popen(
                    ["python", "app.py", "--windows_host_url", "localhost:8006", "--omniparser_server_url", "localhost:8000"],
                    cwd=gradio_path
                )
            return True
        except Exception as e:
            logger.error(f"启动Gradio界面时出错: {e}")
            return False

    def check_service_status(self, url, service_name, timeout=300):
        """检查服务是否正常运行"""
        start_time = time.time()
        retry_count = 0
        max_retries = 3
        
        while time.time() - start_time < timeout:
            try:
                # 尝试不同的端点
                endpoints = ['/', '/health', '/status', '/docs']
                for endpoint in endpoints:
                    try:
                        response = requests.get(f"{url}{endpoint}", timeout=5)
                        if response.status_code in [200, 404]:  # 404也是正常的，因为可能只是没有这个端点
                            logger.info(f"{service_name} 服务已就绪 (尝试访问 {endpoint})")
                            return True
                    except requests.exceptions.RequestException:
                        continue
                
                # 如果所有端点都失败，等待后重试
                retry_count += 1
                if retry_count >= max_retries:
                    logger.warning(f"{service_name} 服务可能未完全启动，继续等待...")
                    retry_count = 0
                
                logger.info(f"等待 {service_name} 服务启动... (已等待 {int(time.time() - start_time)} 秒)")
                time.sleep(5)
                
            except requests.exceptions.ConnectionError:
                logger.info(f"等待 {service_name} 服务启动... (已等待 {int(time.time() - start_time)} 秒)")
                time.sleep(5)
            except Exception as e:
                logger.error(f"检查 {service_name} 服务状态时出错: {e}")
                time.sleep(5)
        
        logger.error(f"{service_name} 服务启动超时")
        return False

    def start_all(self):
        """启动所有服务"""
        if not self.check_conda_env():
            return False

        # 启动OmniParser服务器
        if not self.start_omniparser_server():
            return False
        if not self.check_service_status(self.omniparser_server_url, "OmniParser服务器"):
            return False

        # 启动OmniBox
        if not self.start_omnibox():
            return False
        if not self.check_service_status(self.omnibox_url, "OmniBox"):
            return False

        # 启动Gradio界面
        if not self.start_gradio():
            return False
        if not self.check_service_status(self.gradio_url, "Gradio界面"):
            return False

        logger.info("所有服务已成功启动！")
        logger.info(f"请访问Gradio界面: {self.gradio_url}")
        return True

if __name__ == "__main__":
    starter = OmniToolStarter()
    starter.start_all() 