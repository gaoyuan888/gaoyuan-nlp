# 服务启动脚本

# 这个地方位置随便写，启动后会被选中的基础镜像地址替换
FROM base_image

# 拷贝目录，jd规定不能放在/export下
COPY bin /opt/bin
COPY nlp /opt/nlp

# 执行权限
RUN chmod +x /opt/bin/start.sh

# 执行点，容器启动的时候执行的文件，也是整个服务跑起来的执行点，sleep 9999999d很重要，不能少！
ENTRYPOINT /opt/bin/start.sh && sleep 9999999d
