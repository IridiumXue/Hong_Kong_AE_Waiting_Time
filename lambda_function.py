import json
import csv
import io
import os
import logging
from datetime import datetime, timezone, timedelta
import requests
from huggingface_hub import HfApi

# UTC+8
CHINA_TZ = timezone(timedelta(hours=8))

# 日志配置
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 配置 - 使用环境变量
API_URL = os.environ.get('API_URL', "")
HF_TOKEN = os.environ.get('HF_TOKEN', "")
HF_REPO_ID = os.environ.get('HF_REPO_ID', "")

def get_china_time():
    """获取东八区（中国）当前时间"""
    return datetime.now(CHINA_TZ)

def fetch_hospital_data():
    """从API获取医院等待时间数据"""
    try:
        response = requests.get(API_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"成功从 {API_URL} 获取数据")
            if data.get("success") == "Y" and "result" in data and "hospData" in data["result"]:
                return data["result"]["hospData"]
        logger.error(f"获取数据失败: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"获取数据时出错: {str(e)}")
    return None

def create_csv_content(data):
    """从数据中提取所需字段并创建CSV内容"""
    if not data:
        return None
    
    csv_file = io.StringIO()
    csv_writer = csv.writer(csv_file)
    
    # 标题行
    csv_writer.writerow(["hospCode", "hospTimeEn", "topWait"])
    
    # 数据行
    for hospital in data:
        if "hospCode" in hospital and "topWait" in hospital and "hospTimeEn" in hospital:
            csv_writer.writerow([
                hospital["hospCode"],
                hospital["hospTimeEn"],
                hospital["topWait"]
            ])
    
    csv_content = csv_file.getvalue()
    csv_file.close()
    return csv_content

def upload_to_huggingface(csv_content):
    """将数据上传到HuggingFace（无需保存令牌）"""
    if not csv_content or not HF_TOKEN:
        logger.error("无法上传到HuggingFace: 缺少内容或令牌")
        return False
    
    # 直接使用令牌初始化API，而不调用login()函数
    try:
        hf_api = HfApi(token=HF_TOKEN)
        logger.info("已初始化HuggingFace API客户端")
        
        # 使用东八区时间生成文件名
        now = get_china_time()
        filename = now.strftime("%Y%m%d_%H%M") + ".csv"
        
        # 上传文件
        hf_api.upload_file(
            path_or_fileobj=csv_content.encode('utf-8'),
            path_in_repo=f"data/{filename}",
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        logger.info(f"成功上传 {filename} 到 {HF_REPO_ID}")
        return True
    except Exception as e:
        logger.error(f"上传到HuggingFace时出错: {str(e)}")
        return False

def lambda_handler(event, context):
    """Lambda 函数入口点"""
    now = get_china_time()
    logger.info(f"Lambda 执行开始 (UTC+8: {now.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # 获取数据
    data = fetch_hospital_data()
    if not data:
        return {
            'statusCode': 500,
            'body': json.dumps('获取数据失败')
        }
    
    # 创建CSV内容
    csv_content = create_csv_content(data)
    if not csv_content:
        return {
            'statusCode': 500,
            'body': json.dumps('处理数据失败')
        }
    
    # 上传到HuggingFace
    hf_success = upload_to_huggingface(csv_content)
    
    logger.info("Lambda 执行完成")
    
    return {
        'statusCode': 200 if hf_success else 500,
        'body': json.dumps({
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S %z'),
            'huggingface_upload': hf_success,
            'hospitals_count': len(data)
        })
    }