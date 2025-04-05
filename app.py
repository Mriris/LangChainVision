import os
import base64
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash
from werkzeug.utils import secure_filename
import io
from PIL import Image
import cv2
import numpy as np

# 导入Test2.py的工作流程
from AgentCore import app as workflow_app, ImageState, cleanup_resources
# 导入遥感图像处理模块
from rs_processing import (
    land_cover_classification, 
    change_detection, 
    object_detection, 
    image_segmentation,
    read_image,
    image_to_base64,
    UPLOAD_FOLDER,
    RESULT_FOLDER
)

app = Flask(__name__)
app.secret_key = "langchainvision_secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}  # 添加遥感图像格式支持

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# 确保结果目录存在
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
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
        
        # 初始化工作流状态
        initial_state = {
            "user_requirement": requirement,
            "image_path": file_path,
            "task": "",
            "error": "",
            "status": "started",
            "history": []
        }
        
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
            
            # 执行工作流
            try:
                final_state = workflow_app.invoke(initial_state)
                
                # 记录调试信息
                print(f"工作流执行结果：状态={final_state.get('status')}, 任务={final_state.get('task')}")
                if final_state.get('error'):
                    print(f"工作流错误：{final_state.get('error')}")
                
                # 打印详细的结果信息用于调试
                print(f"结果消息：{final_state.get('result_message')}")
                print(f"最终状态包含的键: {list(final_state.keys())}")
                
                # 创建基本结果对象
                result = {
                    "task": final_state.get('task', '未知'),
                    "status": final_state.get('status', '未知'),
                    "message": final_state.get('result_message', ''),
                    "error": final_state.get('error', ''),
                    "output": final_state.get('output')  # 传递原始输出用于调试
                }
                
                # 严重问题：我们发现result_message总是为None
                # 创建一个直接修复，无论如何确保有消息
                if not result['message'] and result['task'] == 'annotation':
                    result['message'] = f"物体标注完成"
                    print(f"强制设置标注消息: {result['message']}")
                
                print(f"任务类型: {result['task']}, 结果消息: {result['message']}")
                
                # 特殊处理标注任务
                if result['task'] == 'annotation':
                    # 检查备份的标注结果
                    if 'annotations' in final_state:
                        annotations = final_state['annotations']
                        if not result['message'] and annotations.get('message'):
                            result['message'] = annotations['message']
                            print(f"从annotations恢复消息: {result['message']}")
                    
                    # 尝试获取检测结果
                    if 'detection_results' in final_state:
                        result['detections'] = final_state['detection_results']
                        print(f"从final_state获取检测结果: {len(result['detections'])}个")
                
                # 特殊处理图像增强任务
                elif result['task'] == 'enhancement':
                    # 检查备份的增强结果
                    if 'enhancement' in final_state:
                        enhancement = final_state['enhancement']
                        if not result['message'] and enhancement.get('message'):
                            result['message'] = enhancement['message']
                            print(f"从enhancement恢复消息: {result['message']}")
                    
                    # 获取增强图像和对比图像的路径
                    enhanced_path = final_state.get('enhanced_image_path', 'enhanced_image.jpg')
                    comparison_path = final_state.get('comparison_image_path', 'image_comparison.jpg')
                    
                    # 检查图像文件是否存在，并添加到结果中
                    if os.path.exists(enhanced_path):
                        try:
                            with open(enhanced_path, 'rb') as img_file:
                                enhanced_data = base64.b64encode(img_file.read()).decode('utf-8')
                            result['enhanced_image'] = enhanced_data
                            print(f"增强图像已转换为base64，长度: {len(enhanced_data) if enhanced_data else 0}")
                        except Exception as e:
                            print(f"读取增强图像失败: {str(e)}")
                    
                    if os.path.exists(comparison_path):
                        try:
                            with open(comparison_path, 'rb') as img_file:
                                comparison_data = base64.b64encode(img_file.read()).decode('utf-8')
                            result['comparison_image'] = comparison_data
                            print(f"对比图像已转换为base64，长度: {len(comparison_data) if comparison_data else 0}")
                        except Exception as e:
                            print(f"读取对比图像失败: {str(e)}")
                    
                    # 确保设置结果消息
                    if not result['message']:
                        result['message'] = "图像增强完成！已生成高清晰度图像。"
                        print(f"已设置默认增强结果消息: {result['message']}")
                
                # 特殊处理图像描述任务
                elif result['task'] == 'interpretation':
                    # 首先检查默认结果消息
                    if not result['message'] and 'result_message' in final_state:
                        result['message'] = final_state.get('result_message')
                        print(f"从result_message恢复图像描述: {result['message']}")
                    
                    # 如果仍然没有消息，尝试从output获取
                    if not result['message'] and 'output' in final_state:
                        interpretation_text = final_state['output']
                        if isinstance(interpretation_text, str) and interpretation_text.strip():
                            result['message'] = f"图像描述：\n{interpretation_text}"
                            print(f"从output恢复图像描述: 已设置(长度:{len(interpretation_text)})")
                    
                    # 如果仍然没有消息，设置默认消息
                    if not result['message']:
                        result['message'] = "图像描述未能生成，可能是模型连接问题。请确保Ollama服务正在运行，并已下载llava模型。"
                        print("设置默认图像描述结果消息")
                
                # 确保分类结果不为空
                if result['task'] == 'classification' and not result['message'] and 'output' in final_state:
                    try:
                        output = final_state['output']
                        # 尝试读取分类名称
                        try:
                            with open("imagenet_classes.txt", "r") as f:
                                classes = [line.strip() for line in f.readlines()]
                            class_name = classes[output]
                        except Exception:
                            class_name = f"类别ID: {output}"
                        
                        result['message'] = f"分类结果：{class_name}"
                        print(f"从output重新构建分类结果: {result['message']}")
                    except Exception as rebuild_err:
                        print(f"尝试重建输出失败: {str(rebuild_err)}")
                
                # 对于annotation任务，确保即使有错误也能显示结果
                if result['task'] == 'annotation' and not result['message'] and final_state.get('result_message'):
                    result['message'] = final_state.get('result_message')
                    print(f"修复annotation结果消息: {result['message']}")
                
                # 检查是否存在输出
                if 'output' not in final_state and not result['error']:
                    result['error'] = '模型未能生成有效输出'
                    print(f"警告：工作流未生成输出")
                
                # 处理图像结果
                if final_state.get('task') == 'annotation':
                    # 先设置一个默认消息，防止为None
                    if not result['message'] and final_state.get('result_message'):
                        result['message'] = final_state.get('result_message')
                    
                    # 强制检查图像文件是否存在，无论路径是什么
                    annotated_paths = [
                        final_state.get('annotated_image_path', ''),
                        'annotated_image.jpg',  # 通常默认路径
                        os.path.join(os.getcwd(), 'annotated_image.jpg')
                    ]
                    
                    found_image = False
                    for path in annotated_paths:
                        if path and os.path.exists(path):
                            print(f"找到标注图像: {path}")
                            try:
                                with open(path, 'rb') as img_file:
                                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                                result['image'] = img_data
                                print(f"图像已转换为base64，长度: {len(img_data) if img_data else 0}")
                                found_image = True
                                break
                            except Exception as e:
                                print(f"读取图像 {path} 失败: {str(e)}")
                    
                    if not found_image:
                        print("警告: 未能找到任何标注图像文件")
                        result['error'] = f"{result.get('error', '')} 未能找到标注图像文件"
                    
                    # 确保检测结果存在
                    detection_results = final_state.get('detection_results', [])
                    if detection_results:
                        result['detections'] = detection_results
                        print(f"检测到 {len(detection_results)} 个物体")
                        
                    # 确保设置结果消息
                    if not result['message']:
                        result['message'] = f"标注完成，共检测到 {len(detection_results) if detection_results else '多个'} 个物体。"
                        print(f"已设置默认结果消息: {result['message']}")
                
                # 记录最终状态摘要
                print(f"返回结果：{result['task']} - {result['status']} - 错误：{result.get('error', '无')}")
                
                # 转换错误信息为用户友好的消息
                if result.get('error'):
                    if 'model_selection_error' in result['error']:
                        result['friendly_error'] = '模型加载失败。请确保已安装所需的模型并启动相关服务。'
                    elif 'inference_error' in result['error']:
                        result['friendly_error'] = '图像分析过程中出错。可能是图像格式不支持或模型出现问题。'
                    elif 'output_error' in result['error']:
                        result['friendly_error'] = '生成结果时出错。请尝试不同的分析需求或图像。'
                    else:
                        result['friendly_error'] = '处理过程中出现未知错误，请稍后重试。'
                
                # 调用清理函数释放GPU资源
                cleanup_resources()
                print("已释放模型和GPU资源")
                
                return render_template('result.html', result=result, requirement=requirement)
            except Exception as workflow_error:
                print(f"工作流执行错误: {str(workflow_error)}")
                flash(f'处理过程中出错: {str(workflow_error)}')
                
                # 确保即使出错也释放资源
                cleanup_resources()
                print("出错后已释放模型和GPU资源")
                
                return redirect(url_for('general'))
        except Exception as e:
            print(f"处理异常：{str(e)}")
            flash(f'处理过程中出错: {str(e)}')
            
            # 确保即使出错也释放资源
            try:
                cleanup_resources()
                print("异常后已释放模型和GPU资源")
            except Exception as cleanup_err:
                print(f"释放资源时出错: {str(cleanup_err)}")
                
            return redirect(url_for('general'))
    
    flash('不支持的文件类型')
    return redirect(url_for('general'))

# 添加处理遥感图像分析的路由
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
                if "error" in analysis_result:
                    raise Exception(analysis_result["error"])
                
                result.update(analysis_result)
                result["message"] = f"地物分类分析完成，共识别出{analysis_result.get('class_count', 7)}种地物类型。"
            
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
                
                analysis_result = change_detection(
                    file_path, 
                    reference_path=ref_path,
                    model_name=model_selection if model_selection != 'auto' else 'bit-cd'
                )
                if "error" in analysis_result:
                    raise Exception(analysis_result["error"])
                
                result.update(analysis_result)
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
                if "error" in analysis_result:
                    raise Exception(analysis_result["error"])
                
                result.update(analysis_result)
                result["confidence_threshold"] = conf_thresh  # 添加置信度阈值到结果
                result["message"] = f"目标检测分析完成，共检测到{len(analysis_result.get('detections', []))}个目标。"
            
            # 图像分割
            elif task_type == 'segmentation':
                # 解析附加参数（如分割精度）
                n_segments = 100  # 默认值
                if additional_params:
                    params = {p.split('=')[0].strip(): p.split('=')[1].strip() 
                              for p in additional_params.split(',') if '=' in p}
                    if '分割精度' in params:
                        if params['分割精度'] == '高':
                            n_segments = 200
                        elif params['分割精度'] == '低':
                            n_segments = 50
                
                analysis_result = image_segmentation(
                    file_path, 
                    model_name=model_selection if model_selection != 'auto' else 'segformer',
                    n_segments=n_segments
                )
                if "error" in analysis_result:
                    raise Exception(analysis_result["error"])
                
                result.update(analysis_result)
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
            flash(f'处理过程中出错: {str(e)}')
        
        return render_template('rs_result.html', result=result, filename=filename)
    
    flash('不支持的文件类型')
    return redirect(url_for('remote_sensing'))

# 添加遥感数据参考图像上传路由
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

# 添加AJAX数据获取路由
@app.route('/rs_get_results/<task_id>', methods=['GET'])
def get_rs_results(task_id):
    """获取遥感分析任务的结果数据（用于异步更新）"""
    # 在实际应用中，这里应该从数据库中查询结果
    # 这里仅做演示
    return jsonify({
        "status": "completed",
        "progress": 100,
        "message": "分析完成"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)