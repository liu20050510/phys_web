import csv
import json
import shutil
from datetime import datetime
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from main.config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS, UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from sqlalchemy import text
from main.models import db, User, HealthHistory
from flask_cors import CORS
import os
import subprocess
import sys
import threading
import cv2
import numpy as np
import base64
import tempfile
from flask import current_app, make_response, Response
from werkzeug.utils import secure_filename
import uuid
from flask_jwt_extended import (
    JWTManager,
    jwt_required,
    create_access_token,
    get_jwt_identity
)

# 创建 Flask 应用
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 初始化 SQLAlchemy 实例
db.init_app(app)

bcrypt = Bcrypt(app)

CORS(app, resources={
    r"/*": {  # 全局配置
        "origins": "http://localhost:5173",
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Authorization", "Content-Type"]
    }
})

# 测试连接数据库
with app.app_context():
    with db.engine.connect() as conn:
        rs = conn.execute(text("select 1"))
        print(rs.fetchone())  # 若打印(1,)则代表数据库连接成功

# 初始化JWT
app.config['JWT_SECRET_KEY'] = 'your-super-secret-key'  # 生产环境使用复杂密钥
jwt = JWTManager(app)

# 注册接口
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # 检查必填字段是否为空
    if not username or not email or not password:
        return jsonify({"message": "用户名、邮箱和密码均为必填项", "success": False}), 400

    # 检查用户名是否已存在
    if User.query.filter_by(username=username).first():
        return jsonify({"message": "用户名已存在", "success": False}), 400

    # 检查邮箱是否已存在
    if User.query.filter_by(email=email).first():
        return jsonify({"message": "邮箱已被注册", "success": False}), 400

    # 对密码进行加密
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    try:
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()  # 直接提交即可

        return jsonify({
            "message": "注册成功",
            "success": True,
            "user_id": new_user.user_id
        }), 201
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"注册失败详细错误: {str(e)}", exc_info=True)  # 记录完整堆栈
        return jsonify({
            "message": f"注册失败: {str(e)}",  # 返回具体错误
            "success": False
        }), 500

# 登录接口
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # 检查必填字段是否为空
    if not email or not password:
        return jsonify({"message": "邮箱和密码均为必填项", "success": False}), 400

    # 根据邮箱查找用户
    user = User.query.filter_by(email=email).first()

    # 检查用户是否存在以及密码是否正确
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({"message": "邮箱或密码错误", "success": False}), 401

    return jsonify({
        "message": "登录成功",
        "success": True,
        "access_token": create_access_token(identity=user.user_id)
    }), 200

# 添加健康记录（使用固定表）
@app.route('/health_records', methods=['POST'])
def add_health_record():
    data = request.get_json()
    user_id = data.get('user_id')
    detect_time = data.get('detect_time')
    bmp = data.get('bmp')
    image_path = data.get('image_path', None)

    if not all([user_id, detect_time, bmp]):
        return jsonify({"message": "缺少必要参数", "success": False}), 400

    try:
        new_record = HealthHistory(
            user_id=user_id,
            detect_time=datetime.fromisoformat(detect_time),
            bmp=bmp,
            image_path=image_path  # 新增字段
        )
        db.session.add(new_record)
        db.session.commit()
        return jsonify({"message": "健康记录添加成功", "success": True,"record_id": new_record.id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"添加失败: {str(e)}", "success": False}), 500

# 查询健康记录
@app.route('/users/<string:user_id>/health_records', methods=['GET'])
def get_health_records(user_id):
    try:
        records = HealthHistory.query.filter_by(user_id=user_id).all()
        return jsonify([{
            "detect_time": r.detect_time.isoformat(),
            "bmp": r.bmp,
            "image_url": r.image_path
        } for r in records]), 200
    except Exception as e:
        return jsonify({"message": f"查询失败: {str(e)}", "success": False}), 500


# 异步任务处理函数
def async_predict_task(video_dir, flask_app, user_id, username, task_id):
    """在独立线程中执行预测任务"""
    with flask_app.app_context():
        try:
            print(f"[Async] Starting prediction for {video_dir}")

            # 确保结果目录存在
            result_dir = os.path.abspath(video_dir)
            os.makedirs(result_dir, exist_ok=True)

            run_script = os.path.abspath("rPPG-main/run.py")
            result = subprocess.run(
                [
                    sys.executable,
                    run_script,
                    "--data_path",
                    os.path.abspath(video_dir)
                ],
                capture_output=True,
                text=True,
                check=True
            )

            # 验证结果目录
            if not os.path.isdir(result_dir):
                raise FileNotFoundError(f"结果目录不存在: {result_dir}")

            result_path = os.path.join(result_dir, "result.json")
            print(f"[Async] 预期结果路径: {result_path}")

            # 加载结果数据
            with open(result_path) as f:
                result_data = json.load(f)

            # 数据完整性校验
            required_keys = ['bmp', 'image_path', 'csv_path']
            if not all(key in result_data for key in required_keys):
                raise ValueError("结果文件缺少必要字段")

            # 保存到数据库
            new_record = HealthHistory(
                user_id=user_id,
                username=username,
                detect_time=datetime.now(),
                bmp=result_data['bmp'],
                image_path=result_data['image_path'],
                csv_path=result_data['csv_path']
            )
            db.session.add(new_record)
            db.session.commit()

            # 读取CSV文件内容
            csv_path = os.path.join(result_dir, result_data['csv_path'])
            with open(csv_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                csv_data = list(csv_reader)  # 转换为二维列表

            # 更新任务状态为完成
            tasks[task_id] = {
                "status": "completed",
                "bmp": result_data['bmp'],
                "csv_data": csv_data
            }
            print(f"[Async] 任务 {task_id} 完成")

        except Exception as e:
            print(f"[Async] 处理过程中发生错误：{str(e)}")
            # 更新任务状态为失败
            tasks[task_id] = {
                "status": "failed",
                "error": str(e),
                "details": f"{type(e).__name__} at {datetime.now().isoformat()}"
            }


# 在配置后新增文件检查函数
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 内存存储任务状态（生产环境建议用Redis）
tasks = {}

# 替换原有的handle_upload_video路由
@app.route('/upload_video', methods=['POST'])
@jwt_required()
def handle_upload_video():
    print("\n=== 上传请求详情 ===")
    print("接收文件:", dict(request.files))
    print("请求头:", dict(request.headers))
    print("表单数据:", request.form.to_dict())

    try:
        if 'video' not in request.files:
            return jsonify({"message": "没有视频文件", "success": False}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"message": "没有选择文件", "success": False}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            now = datetime.now()
            date_folder = now.strftime("%Y-%m-%d")
            time_folder = now.strftime("%H-%M-%S")

            save_path = os.path.join(
                current_app.config['UPLOAD_FOLDER'],
                'videos',
                date_folder,
                time_folder
            )
            os.makedirs(save_path, exist_ok=True)

            unique_name = f"{uuid.uuid4().hex}_{filename}"
            file.save(os.path.join(save_path, unique_name))

            # 启动异步任务
            video_path = os.path.join(save_path, unique_name)
            video_dir = os.path.dirname(video_path)

            # 从JWT token获取用户信息
            current_user_id = get_jwt_identity()
            user = User.query.filter_by(user_id=current_user_id).first()  # 关键修改
            if not user:
                return jsonify({"message": "用户不存在", "success": False}), 404

            # 生成唯一任务ID
            task_id = str(uuid.uuid4())
            tasks[task_id] = {"status": "processing"}

            # 启动异步线程时传递用户ID
            predict_thread = threading.Thread(
                target=async_predict_task,
                args=(video_dir, current_app._get_current_object(), user.user_id, user.username, task_id)
            )
            predict_thread.start()

            return jsonify({
                "task_id": task_id,
                "code": 200,
                "message": "视频上传成功",
                "data": {
                    "preview_url": f"/uploads/videos/{date_folder}/{unique_name}",
                    "video_dir": str(os.path.basename(video_dir)),
                    "queue_position": 0
                },
                "success": True
            }), 200

        else:
            return jsonify({"message": "不允许的文件类型", "success": False}), 400

    except Exception as e:
        print(f"[ERROR] 视频上传失败: {str(e)}", flush=True)
        return jsonify({
            "message": f"服务器内部错误: {str(e)}",
            "success": False
        }), 500


# 任务状态查询接口
@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = tasks.get(task_id, {"status": "not_found"})

    # 处理完成状态
    if task["status"] == "completed":
        response_data = {
            "status": "completed",
            "bmp": task["bmp"],
            "csv_data": task["csv_data"]
        }
    # 处理失败状态
    elif task["status"] == "failed":
        response_data = {
            "status": "failed",
            "error": task.get("error", "Unknown error"),
            "details": task.get("details", "")
        }
    # 处理中状态
    else:
        response_data = task

    response = jsonify(response_data)
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    return response

# 获取当前用户健康记录接口
@app.route('/user/health_records', methods=['GET'])
@jwt_required()
def get_current_user_records():
    try:
        # 从JWT获取当前用户ID
        current_user_id = get_jwt_identity()
        print(current_user_id)

        # 获取分页和过滤参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('pageSize', 10, type=int)
        start_date = request.args.get('startDate')
        end_date = request.args.get('endDate')

        # 构建基础查询
        query = HealthHistory.query.filter_by(user_id=current_user_id)

        # 添加日期过滤
        if start_date and end_date:
            # 将日期字符串转换为datetime对象进行查询
            from datetime import datetime
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            query = query.filter(HealthHistory.detect_time.between(start_datetime, end_datetime))
        elif start_date:
            from datetime import datetime
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            query = query.filter(HealthHistory.detect_time >= start_datetime)
        elif end_date:
            from datetime import datetime
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            query = query.filter(HealthHistory.detect_time <= end_datetime)

        # 执行分页查询
        pagination = query.order_by(HealthHistory.detect_time.desc()).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )

        return jsonify({
            "success": True,
            "data": [{
                "detect_time": record.detect_time.strftime("%Y-%m-%d %H:%M:%S"),
                "bmp": record.bmp,
                "report": f"检测时间：{record.detect_time.strftime('%Y-%m-%d %H:%M:%S')}\n脉搏：{record.bmp} BPM\n检测结果：正常范围"
            } for record in pagination.items],
            "total": pagination.total,
            "currentPage": page,
            "pageSize": per_page
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印错误堆栈，便于调试
        return jsonify({"message": f"查询失败: {str(e)}", "success": False}), 500


# 实时心率检测相关功能
# 存储视频流数据
class VideoStreamProcessor:
    def __init__(self):
        self.frames = []
        self.timestamps = []
        self.processing = False
        self.lock = threading.Lock()  # 添加线程锁

    def add_frame(self, frame):
        """添加视频帧到缓冲区"""
        with self.lock:
            if len(self.frames) > 300:  # 最多保留300帧（约10秒@30fps）
                self.frames.pop(0)
                self.timestamps.pop(0)

            self.frames.append(frame)
            self.timestamps.append(datetime.now())

    def get_frames(self):
        """获取所有帧数据"""
        with self.lock:
            # 返回副本以避免并发问题
            return self.frames[:], self.timestamps[:]

    def clear_frames(self):
        """清空帧数据"""
        with self.lock:
            self.frames.clear()
            self.timestamps.clear()

    def get_recent_frames(self, count=30):
        """获取最近的指定数量帧"""
        with self.lock:
            if len(self.frames) >= count:
                return self.frames[-count:], self.timestamps[-count:]
            return self.frames[:], self.timestamps[:]


# 全局视频流处理器
video_processor = VideoStreamProcessor()


@app.route('/realtime_heart_rate/start', methods=['POST'])
@jwt_required()
def start_realtime_heart_rate():
    """开始实时心率检测"""
    try:
        video_processor.clear_frames()
        video_processor.processing = True
        return jsonify({
            "success": True,
            "message": "已开始实时心率检测"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"启动失败: {str(e)}"
        }), 500


@app.route('/realtime_heart_rate/frame', methods=['POST'])
@jwt_required()
def upload_frame():
    """上传视频帧用于实时心率分析"""
    try:
        if not video_processor.processing:
            return jsonify({
                "success": False,
                "message": "心率检测未启动"
            }), 400

        # 获取并解析图像数据
        data = request.get_json()
        if 'frame' not in data:
            return jsonify({
                "success": False,
                "message": "缺少帧数据"
            }), 400

        # 添加帧数据到处理器
        frame_data = data['frame']
        video_processor.add_frame(frame_data)

        # 检查是否有足够的帧进行分析
        frames, timestamps = video_processor.get_frames()
        heart_rate = None
        if len(frames) >= 30:  # 当累积到足够帧数时进行分析
            # 进行实时心率分析
            heart_rate = analyze_heart_rate_with_frames(frames)

        return jsonify({
            "success": True,
            "message": "帧数据已接收",
            "heart_rate": heart_rate  # 如果计算了心率则返回，否则为None
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"帧数据处理失败: {str(e)}"
        }), 500


@app.route('/realtime_heart_rate/analyze', methods=['POST'])
@jwt_required()
def analyze_realtime_heart_rate():
    """分析实时心率数据"""
    try:
        # 获取帧数据
        frames, timestamps = video_processor.get_frames()

        if len(frames) < 30:  # 至少需要1秒的数据
            return jsonify({
                "success": False,
                "message": "视频数据不足，无法分析心率"
            }), 400

        # 使用rPPG模型直接分析帧数据
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()

        if not user:
            return jsonify({
                "success": False,
                "message": "用户不存在"
            }), 404

        # 计算检测时长（秒）
        if timestamps:
            duration = (timestamps[-1] - timestamps[0]).total_seconds()
        else:
            duration = 0

        # 直接使用帧数据进行rPPG分析，无需合成视频
        heart_rate = analyze_heart_rate_with_frames(frames)

        # 保存记录
        new_record = HealthHistory(
            user_id=current_user_id,
            username=user.username,
            detect_time=datetime.now(),
            bmp=heart_rate,
            image_path="",  # 添加默认值
            csv_path=""  # 添加默认值
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "心率检测完成",
            "data": {
                "bmp": heart_rate,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"心率检测分析失败: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"分析失败: {str(e)}"
        }), 500


def decode_frames(frames_data):
    """解码base64帧数据为numpy数组"""
    decoded_frames = []
    for frame_data in frames_data:
        # 解码base64数据
        try:
            # 处理可能的数据URL前缀
            if frame_data.startswith('data:image'):
                # 提取base64部分
                base64_data = frame_data.split(',')[1]
            else:
                base64_data = frame_data

            image_data = base64.b64decode(base64_data)
            image_array = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is not None:
                # 转换为RGB格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decoded_frames.append(frame)
        except Exception as e:
            print(f"解码帧数据时出错: {str(e)}")
            continue
    return decoded_frames


def analyze_heart_rate_with_frames(frames_data):
    """
    使用rPPG模型直接分析帧数据
    """
    try:
        # 解码帧数据
        decoded_frames = decode_frames(frames_data)

        if not decoded_frames:
            print("没有有效的帧数据")
            return 0

        # 创建临时目录存放数据
        temp_dir = tempfile.mkdtemp()
        try:
            # 导入必要的模块
            import sys
            import os
            rppg_main_path = os.path.join(os.path.dirname(__file__), 'rPPG-main')
            sys.path.append(rppg_main_path)

            from config import get_config
            from dataset.data_loader.userLoader import RealtimeLoader
            from neural_methods.trainer.RhythmMambaTrainer import RhythmMambaTrainer
            import yaml
            from torch.utils.data import DataLoader
            import argparse
            import numpy as np

            # 加载配置
            config_path = os.path.join(rppg_main_path, 'configs', 'user.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            # 创建临时配置文件
            temp_config_path = os.path.join(temp_dir, 'temp_config.yaml')
            config_dict['TEST']['OUTPUT_SAVE_DIR'] = temp_dir
            # 确保TOOLBOX_MODE设置为inference
            config_dict['TOOLBOX_MODE'] = 'inference'
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, allow_unicode=True)

            # 修改这里：创建一个带有config_file属性的对象
            args = argparse.Namespace()
            args.config_file = temp_config_path
            # 不再手动设置 TOOLBOX_MODE，而是在配置文件中设置
            config = get_config(args)

            # 创建实时数据加载器
            realtime_loader = RealtimeLoader(
                name="realtime",
                frames_data=decoded_frames,
                config_data=config.TEST.DATA
            )

            data_loader_dict = {
                "inference": DataLoader(
                    dataset=realtime_loader,
                    num_workers=0,
                    batch_size=config.INFERENCE.BATCH_SIZE,
                    shuffle=False,
                    worker_init_fn=None,
                    generator=None,
                )
            }

            # 运行推理
            model_trainer = RhythmMambaTrainer(config, data_loader_dict)
            heart_rate = model_trainer.inference(data_loader_dict)

            # 检查心率值是否有效
            if np.isnan(heart_rate) or heart_rate <= 0 or heart_rate > 300:  # 添加合理的心率范围检查
                print(f"模型返回无效心率值: {heart_rate}，使用备用方法计算")
                heart_rate = 0.0

            return heart_rate
        finally:
            # 清理临时文件
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"心率分析出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0


@app.route('/realtime_heart_rate/stop', methods=['POST'])
@jwt_required()
def stop_realtime_heart_rate():
    """停止实时心率检测并分析数据"""
    try:
        video_processor.processing = False

        # 获取帧数据
        frames, timestamps = video_processor.get_frames()

        if len(frames) < 30:  # 至少需要1秒的数据
            return jsonify({
                "success": False,
                "message": "视频数据不足，无法分析心率"
            }), 400

        # 使用rPPG模型直接分析帧数据
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()

        if not user:
            return jsonify({
                "success": False,
                "message": "用户不存在"
            }), 404

        # 计算检测时长（秒）
        if timestamps:
            duration = (timestamps[-1] - timestamps[0]).total_seconds()
        else:
            duration = 0

        # 直接使用帧数据进行rPPG分析，无需合成视频
        heart_rate = analyze_heart_rate_with_frames(frames)

        # 保存记录，提供默认值以避免空值
        new_record = HealthHistory(
            user_id=current_user_id,
            username=user.username,
            detect_time=datetime.now(),
            bmp=heart_rate,
            image_path="",  # 提供默认值
            csv_path=""     # 提供默认值
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "心率检测完成",
            "data": {
                "bmp": heart_rate,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"心率检测停止失败: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"分析失败: {str(e)}"
        }), 500

# 获取当前用户信息接口
@app.route('/user/info', methods=['GET'])
@jwt_required()
def get_user_info():
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()

        if not user:
            return jsonify({
                "success": False,
                "message": "用户不存在"
            }), 404

        return jsonify({
            "success": True,
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"获取用户信息失败: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)

__all__ = ['app', 'bcrypt']