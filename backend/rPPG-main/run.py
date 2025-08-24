# run.py
import subprocess
import sys
import os
import yaml
import re
import argparse
import tempfile
import shutil

def parse_args():
    """新增命令行参数解析"""
    parser = argparse.ArgumentParser(description='rPPG视频处理管道')
    parser.add_argument('--data_path', type=str,
                       help='覆盖配置文件中的DATA_PATH路径',
                       default=None)
    return parser.parse_args()

def get_video_fps(video_path):
    """使用ffprobe获取视频帧率"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()
        # 处理分数形式的帧率（如30000/1001）
        if '/' in fps_str:
            numerator, denominator = map(int, fps_str.split('/'))
            return round(numerator / denominator, 2)
        return float(fps_str)
    except subprocess.CalledProcessError as e:
        print(f"获取帧率失败: {e}")
        return None

def convert_to_30fps(input_path, output_path):
    """使用ffmpeg转换视频为30fps"""
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖已存在文件
        '-i', input_path,
        '-r', '30',  # 设置输出帧率
        '-c:v', 'libx264',  # 使用H.264编码
        '-preset', 'fast',  # 编码预设
        '-crf', '23',  # 质量参数
        '-pix_fmt', 'yuv420p',  # 像素格式
        output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")
        return False


def process_videos(config):
    """修改为直接接收配置字典"""
    data_path = config['TEST']['DATA']['DATA_PATH']
    if not os.path.exists(data_path):
        print(f"数据路径不存在: {data_path}")
        return

    for filename in os.listdir(data_path):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_path = os.path.join(data_path, filename)
            temp_path = os.path.join(data_path, f"temp_{filename}")

            fps = get_video_fps(input_path)
            if fps is None:
                continue

            if abs(fps - 30.0) > 0.01:  # 允许微小误差
                print(f"转换 {filename} 从 {fps}FPS 到 30FPS")
                if convert_to_30fps(input_path, temp_path):
                    # 替换原始文件
                    os.replace(temp_path, input_path)
            else:
                print(f"{filename} 已经是30FPS，跳过转换")

def create_temp_config(original_path, new_data_path):
    """创建临时配置文件"""
    with open(original_path, 'r') as f:
        config = yaml.safe_load(f)

    # 更新数据路径
    if new_data_path:
        abs_data_path = os.path.abspath(new_data_path)
        config['TEST']['DATA']['DATA_PATH'] = abs_data_path
        # 关键修改：动态设置输出目录为视频所在目录
        config['TEST']['OUTPUT_SAVE_DIR'] = abs_data_path  # 新增此行

    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp_config.yaml")

    with open(temp_path, 'w') as f:
        yaml.dump(config, f)

    return temp_path, temp_dir

def main():
    args = parse_args()

    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))  # run.py所在目录
    rppg_root = os.path.dirname(current_dir)  # 项目根目录

    # 创建临时配置文件
    original_config_path = os.path.join(
        current_dir,  # 使用run.py所在目录
        "configs",
        "user.yaml"
    )
    temp_config_path, temp_dir = create_temp_config(
        original_config_path,
        args.data_path
    )

    try:
        # 加载临时配置用于视频处理
        with open(temp_config_path, 'r') as f:
            temp_config = yaml.safe_load(f)

        # 处理视频文件
        process_videos(temp_config)

        # 运行主程序
        command = [
            sys.executable,
            os.path.join(current_dir, "main.py"),  # 使用绝对路径
            "--config_file",
            temp_config_path
        ]
        result = subprocess.run(command, check=True, cwd=current_dir)
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()