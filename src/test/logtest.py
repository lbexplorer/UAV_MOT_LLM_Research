import logging
# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为 DEBUG，确保 DEBUG、INFO 等日志都可以输出
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.debug("这是一条调试信息")
logging.info("这是一条信息级别的日志")
logging.warning("这是一条警告级别的日志")
logging.error("这是一条错误级别的日志")
logging.critical("这是一条严重级别的日志")