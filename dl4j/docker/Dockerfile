FROM ubuntu
LABEL Description="This image is ubuntu with dl4j 1.0.0_alpha." Vendor="liweigu" Version="1.0"
COPY cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb /soft/cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb
COPY installCuda.sh /soft/installCuda.sh
COPY OpenBLAS /soft/OpenBLAS
COPY gcc-trunk /opt/gcc-trunk
COPY l_mkl_2018.2.199 /soft/l_mkl_2018.2.199
COPY start.sh /soft/l_mkl_2018.2.199/start.sh
COPY locales /var/lib/locales
COPY start.py /python/start.py
RUN apt-get -y update && \
	apt-get install -y pciutils gcc linux-headers-$(uname -r) default-jre tzdata gcc automake autoconf libtool make tcl tk expect cpio locales fonts-droid-fallback ttf-wqy-zenhei ttf-wqy-microhei fonts-arphic-ukai fonts-arphic-uming python && \
	dpkg -i cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb && \
	apt-key add /var/cuda-repo-9-1-local/7fa2af80.pub && \
	apt-get -y update && \
	expect installCuda.sh && \
	ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
	cd /soft/OpenBLAS && make && make PREFIX=/usr/local/ install && \
	cd /soft/l_mkl_2017.3.196/ && expect start.sh && \
	locale-gen zh_CN.UTF-8
ENV LC_ALL="zh_CN.utf8" TZ="Asia/Shanghai" LANG="C.UTF-8" LD_LIBRARY_PATH="/opt/gcc-trunk/lib64:$LD_LIBRARY_PATH:/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:/soft/so" PATH="/opt/intel/bin:/opt/gcc-trunk/bin:$PATH"
ENV PATH /usr/local/cuda-9.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH /usr/local/cuda-9.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
CMD python /python/start.py
