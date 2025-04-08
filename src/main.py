import os
import base64
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash
from werkzeug.utils import secure_filename
import io
from PIL import Image
import cv2
import numpy as np
import random

from src.agents.workflow import execute_workflow, workflow_app
from src.utils.image_utils import image_to_base64, read_image
from src.config.settings import (
    UPLOAD_FOLDER, 
    RESULT_FOLDER, 
    ALLOWED_EXTENSIONS
)
from src.rs_processing import (
    land_cover_classification, 
    change_detection, 
    object_detection, 
    image_segmentation
)

# 创建Flask应用
app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.secret_key = "langchainvision_secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    """主页，提供导航到通用图像分析或遥感图像分析"""
    return render_template('index.html')

@app.route('/general')
def general():
    """通用图像分析页面"""
    return render_template('general.html')

@app.route('/remote_sensing')
def remote_sensing():
    """遥感图像分析页面"""
    return render_template('remote_sensing.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理图像上传和分析请求"""
    # 检查是否有文件和需求
    if 'file' not in request.files:
        flash('没有选择文件')
        return redirect(request.url)
    
    file = request.files['file']
    requirement = request.form.get('requirement', '')
    
    if file.filename == '':
        flash('没有选择文件')
        return redirect(request.url)
    
    if not requirement:
        flash('请输入分析需求')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # 保存上传的文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # 检查Ollama服务是否可用
            try:
                from langchain_ollama import OllamaLLM
                test_model = OllamaLLM(
                    model="phi4-mini:latest", 
                    base_url="http://localhost:11434"
                )
                test_result = test_model.invoke("测试Ollama连接")
                print("Ollama服务可用")
            except Exception as ollama_err:
                print(f"Ollama服务检测失败: {str(ollama_err)}")
                flash('Ollama服务未启动或不可用。请确保已启动Ollama并下载所需模型。')
                return redirect(url_for('general'))
            
            # 执行图像分析工作流
            print(f"开始处理图像: {file_path}, 需求: {requirement}")
            
            # 使用重构后的工作流执行
            final_state = execute_workflow(file_path, requirement)
            
            # 创建结果对象
            result = {
                "task": final_state.get('task', '未知'),
                "status": final_state.get('status', '未知'),
                "message": final_state.get('result_message', ''),
                "error": final_state.get('error', '')
            }
            
            # 处理特定任务的结果
            if result['task'] == 'annotation':
                # 如果状态中有检测结果，添加到结果中
                if 'detection_results' in final_state:
                    result['detections'] = final_state['detection_results']
                
                # 添加标注图像
                if 'annotated_image_path' in final_state and os.path.exists(final_state['annotated_image_path']):
                    result['image'] = image_to_base64(final_state['annotated_image_path'])
                
            # 处理图像增强结果
            elif result['task'] == 'enhancement':
                # 添加增强图像
                if 'enhanced_image_path' in final_state and os.path.exists(final_state['enhanced_image_path']):
                    result['enhanced_image'] = image_to_base64(final_state['enhanced_image_path'])
                
                # 添加对比图像
                if 'comparison_image_path' in final_state and os.path.exists(final_state['comparison_image_path']):
                    result['comparison_image'] = image_to_base64(final_state['comparison_image_path'])
            
            # 如果有错误信息，创建用户友好的错误消息
            if result.get('error'):
                if 'model_selection_error' in result['error']:
                    result['friendly_error'] = '模型加载失败。请确保已安装所需的模型并启动相关服务。'
                elif 'inference_error' in result['error']:
                    result['friendly_error'] = '图像分析过程中出错。可能是图像格式不支持或模型出现问题。'
                elif 'output_error' in result['error']:
                    result['friendly_error'] = '生成结果时出错。请尝试不同的分析需求或图像。'
                else:
                    result['friendly_error'] = '处理过程中出现未知错误，请稍后重试。'
            
            return render_template('result.html', result=result, requirement=requirement)
            
        except Exception as workflow_error:
            print(f"工作流执行错误: {str(workflow_error)}")
            flash(f'处理过程中出错: {str(workflow_error)}')
            return redirect(url_for('general'))
    
    flash('不支持的文件类型')
    return redirect(url_for('general'))

@app.route('/rs_upload', methods=['POST'])
def rs_upload_file():
    """处理遥感图像上传和分析"""
    # 检查是否有文件和需求
    if 'file' not in request.files:
        flash('没有选择文件')
        return redirect(url_for('remote_sensing'))
    
    file = request.files['file']
    task_type = request.form.get('task_type', '')
    
    if file.filename == '':
        flash('没有选择文件')
        return redirect(url_for('remote_sensing'))
    
    if not task_type:
        flash('请选择任务类型')
        return redirect(url_for('remote_sensing'))
    
    # 获取可选参数
    model_selection = request.form.get('model_selection', 'auto')
    additional_params = request.form.get('additional_params', '')
    
    if file and allowed_file(file.filename):
        # 保存上传的文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 根据任务类型调用相应的分析函数
        result = {"task": task_type, "status": "completed", "is_remote_sensing": True}
        
        try:
            # 地物分类
            if task_type == 'land_cover':
                analysis_result = land_cover_classification(
                    file_path, 
                    model_name=model_selection if model_selection != 'auto' else 'swin-t'
                )
                if isinstance(analysis_result, dict) and "error" in analysis_result:
                    raise Exception(analysis_result["error"])
                
                result.update(analysis_result)
                # 确保classes存在
                if "classes" not in result:
                    result["classes"] = {
                        "0": "森林", "1": "灌木", "2": "草地", "3": "裸地", 
                        "4": "水体", "5": "建筑", "6": "农田"
                    }
                # 确保stats包含color属性
                if "stats" in result:
                    for class_id, stats in result["stats"].items():
                        if "color" not in stats:
                            color_mapping = {
                                "0": [0, 100, 0],     # 深绿色 - 森林
                                "1": [124, 252, 0],   # 浅绿色 - 灌木
                                "2": [152, 251, 152], # 淡绿色 - 草地
                                "3": [139, 69, 19],   # 棕色 - 裸地
                                "4": [0, 0, 255],     # 蓝色 - 水体
                                "5": [128, 128, 128], # 灰色 - 建筑
                                "6": [255, 255, 0],   # 黄色 - 农田
                            }
                            default_color = [200, 200, 200]  # 默认灰色
                            stats["color"] = color_mapping.get(str(class_id), default_color)
                
                result["message"] = f"地物分类分析完成，共识别出{result.get('class_count', 7)}种地物类型。"
            
            # 变化检测
            elif task_type == 'change_detection':
                # 检查是否上传了参考图像
                if 'reference_file' in request.files and request.files['reference_file'].filename != '':
                    ref_file = request.files['reference_file']
                    ref_filename = secure_filename(ref_file.filename)
                    ref_path = os.path.join(app.config['UPLOAD_FOLDER'], f"ref_{ref_filename}")
                    ref_file.save(ref_path)
                else:
                    ref_path = None
                
                # 确定使用的检测方法
                method = "deep"  # 默认方法
                if model_selection == "bit-cd":
                    method = "bitemporal"
                elif model_selection == "changeformer":
                    method = "siamese"
                
                analysis_result = change_detection(
                    file_path, 
                    reference_path=ref_path,
                    method=method
                )
                if isinstance(analysis_result, dict) and "error" in analysis_result:
                    raise Exception(analysis_result["error"])
                
                result.update(analysis_result)
                # 确保change_types存在
                if "change_types" not in result:
                    result["change_types"] = {
                        0: "无变化", 1: "轻微变化", 2: "中等变化", 
                        3: "强烈变化", 4: "水域扩张", 5: "植被增加"
                    }
                result["message"] = "变化检测分析完成，已识别出多种地表变化类型。"
            
            # 目标识别
            elif task_type == 'object_detection':
                # 解析附加参数（如检测阈值）
                conf_thresh = 0.5  # 默认值
                if additional_params:
                    # 解析类似"检测置信度=0.6"这样的参数
                    params = {p.split('=')[0].strip(): p.split('=')[1].strip() 
                              for p in additional_params.split(',') if '=' in p}
                    if '检测置信度' in params:
                        try:
                            conf_thresh = float(params['检测置信度'])
                        except ValueError:
                            pass
                
                analysis_result = object_detection(
                    file_path, 
                    model_name=model_selection if model_selection != 'auto' else 'yolov8-rs',
                    conf_thresh=conf_thresh
                )
                if isinstance(analysis_result, dict) and "error" in analysis_result:
                    raise Exception(analysis_result["error"])
                
                result.update(analysis_result)
                result["confidence_threshold"] = conf_thresh  # 添加置信度阈值到结果
                # 确保class_counts存在
                if "class_counts" not in result:
                    result["class_counts"] = {}
                
                # 确保class_confidences存在
                if "class_confidences" not in result:
                    result["class_confidences"] = {}
                
                # 计算每个类别的数量和平均置信度
                if "detections" in result and result["detections"]:
                    for det in result.get("detections", []):
                        cls = det.get("class", "未知")
                        conf = det.get("confidence", 0.0)
                        
                        # 更新类别计数
                        if cls in result["class_counts"]:
                            result["class_counts"][cls] += 1
                            # 累加置信度，稍后计算平均值
                            result["class_confidences"][cls] = (
                                result["class_confidences"].get(cls, 0.0) + conf
                            )
                        else:
                            result["class_counts"][cls] = 1
                            result["class_confidences"][cls] = conf
                    
                    # 计算平均置信度
                    for cls in result["class_confidences"]:
                        if result["class_counts"][cls] > 0:
                            result["class_confidences"][cls] /= result["class_counts"][cls]
                
                result["message"] = f"目标检测分析完成，共检测到{len(result.get('detections', []))}个目标。"
            
            # 图像分割
            elif task_type == 'segmentation':
                # 解析附加参数（如分割精度）
                precision = "medium"  # 默认值
                if additional_params:
                    params = {p.split('=')[0].strip(): p.split('=')[1].strip() 
                              for p in additional_params.split(',') if '=' in p}
                    if '分割精度' in params:
                        if params['分割精度'] == '高':
                            precision = "high"
                        elif params['分割精度'] == '低':
                            precision = "low"
                
                analysis_result = image_segmentation(
                    file_path, 
                    precision=precision,
                    model_name=model_selection if model_selection != 'auto' else 'unet'
                )
                if isinstance(analysis_result, dict) and "error" in analysis_result:
                    raise Exception(analysis_result["error"])
                
                result.update(analysis_result)
                
                # 确保区域统计格式正确
                if "region_stats" in result:
                    # 确保region_stats是字典格式，键为区域ID，值为区域属性
                    if isinstance(result["region_stats"], list):
                        region_stats_dict = {}
                        for i, region in enumerate(result["region_stats"]):
                            region_id = region.get("id", i)
                            region_stats_dict[str(region_id)] = {
                                "area": region.get("area", 0),
                                "type": region.get("type", "未知"),
                                "brightness": region.get("brightness", 0),
                                "color": region.get("color", [random.randint(0, 255) for _ in range(3)])
                            }
                        result["region_stats"] = region_stats_dict
                    
                    # 确保每个区域都有color属性
                    for region_id, stats in result["region_stats"].items():
                        if "color" not in stats:
                            stats["color"] = [random.randint(0, 255) for _ in range(3)]
                
                result["message"] = "图像分割分析完成，已提取出主要地物边界。"
            
            else:
                result["message"] = f"不支持的任务类型: {task_type}"
                result["status"] = "error"
                return render_template('rs_result.html', result=result, filename=filename)
                
        except Exception as e:
            print(f"遥感分析错误: {str(e)}")
            result["status"] = "error"
            result["error"] = str(e)
            result["message"] = f"分析过程中出错: {str(e)}"
            
            # 确保classes或其他必要字段存在，避免模板渲染错误
            if "classes" not in result and task_type == 'land_cover':
                result["classes"] = {0: "未知"}
            if "change_types" not in result and task_type == 'change_detection':
                result["change_types"] = {0: "无变化"}
            if "class_counts" not in result and task_type == 'object_detection':
                result["class_counts"] = {}
            
            flash(f'处理过程中出错: {str(e)}')
        
        return render_template('rs_result.html', result=result, filename=filename)
    
    flash('不支持的文件类型')
    return redirect(url_for('remote_sensing'))

@app.route('/rs_reference_upload', methods=['POST'])
def rs_reference_upload():
    """处理变化检测的参考图像上传"""
    if 'reference_file' not in request.files:
        return jsonify({"status": "error", "message": "没有选择文件"})
    
    file = request.files['reference_file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "没有选择文件"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"ref_{filename}")
        file.save(file_path)
        
        # 读取图像并转换为Base64用于预览
        image = read_image(file_path)
        if image is None:
            return jsonify({"status": "error", "message": "无法读取图像"})
        
        # 将图像转换为JPEG格式的Base64字符串
        is_success, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        io_buf = io.BytesIO(buffer)
        encoded_img = base64.b64encode(io_buf.getvalue()).decode('utf-8')
        
        return jsonify({
            "status": "success", 
            "filename": filename,
            "preview": encoded_img
        })
    
    return jsonify({"status": "error", "message": "不支持的文件类型"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 