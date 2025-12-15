# # syntax=docker/dockerfile:1.4

# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# # Set default RUN shell to /bin/bash
# SHELL ["/bin/bash", "-cu"]

# ENV http_proxy="http://10.0.0.12:8001"
# ENV https_proxy="http://10.0.0.12:8001"
# ENV ftp_proxy="http://10.0.0.12:8001"

# # Set environment variables
# ENV LANG C.UTF-8
# ENV LC_ALL C.UTF-8

# # Install basic packages for compiling and building
# ENV DEBIAN_FRONTEND=noninteractive
# # change apt source to Tsinghua mirror (optional)
# RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
#     sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
# RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
#     build-essential \
#     cmake \
#     g++-7 \
#     git \
#     curl \
#     wget \
#     ca-certificates \
#     libjpeg-dev \
#     libpng-dev \
#     librdmacm1 \
#     libibverbs1 \
#     libboost-all-dev \
#     ibverbs-providers \
#     tzdata \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     fontconfig \
#     language-pack-en \
#     sysstat \
#     gnupg \
#     lsb-release \
#     sudo \
#     && rm -rf /var/lib/apt/lists/*

#     # Install Miniconda & use Python 3.9
# ARG python=3.9
# ENV PYTHON_VERSION=${python}
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install-conda.sh \
#     && chmod +x /tmp/install-conda.sh \
#     && bash /tmp/install-conda.sh -b -f -p /usr/local \
#     && rm -f /tmp/install-conda.sh \
#     && export HOME=/root PATH=/usr/local/bin:$PATH \
#     && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
#     && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
#     && /usr/local/bin/conda install -y python=${PYTHON_VERSION} \
#     && /usr/local/bin/conda clean -y --all

# # # Setup TUNA mirror (optional)
# # RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# # RUN mkdir -p ~/.conda && \
# #     echo "channels:" > ~/.condarc && \
# #     echo "  - defaults" >> ~/.condarc && \
# #     echo "show_channel_urls: true" >> ~/.condarc && \
# #     echo "channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda" >> ~/.condarc && \
# #     echo "default_channels:" >> ~/.condarc && \
# #     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main" >> ~/.condarc && \
# #     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free" >> ~/.condarc && \
# #     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r" >> ~/.condarc && \
# #     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro" >> ~/.condarc && \
# #     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2" >> ~/.condarc && \
# #     echo "custom_channels:" >> ~/.condarc && \
# #     echo "  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
# #     echo "  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
# #     echo "  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
# #     echo "  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
# #     echo "  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
# #     echo "  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
# #     echo "  nvidia: https://mirrors.sustech.edu.cn/anaconda-extra/cloud" >> ~/.condarc

#     # By default, install packages from `requirements.txt` with pip.
#     COPY . /app
#     # Ensure TOS accepted before creating environments (first.yml)
#     RUN   export HOME=/root PATH=/usr/local/bin:$PATH \
#     && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
#     && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
#     && /usr/local/bin/conda env create -f /app/docker/first.yml \
#     && /usr/local/bin/conda clean -y --all

#     # Ensure RDKit is available in ehignfirst environment (used by preprocessing)
#     RUN export HOME=/root PATH=/usr/local/bin:$PATH \
#         && /usr/local/bin/conda install -n ehignfirst -c conda-forge -y rdkit=2024.09.4 \
#         && /usr/local/bin/conda clean -y --all

#     # Ensure TOS accepted before creating environments (lasttwo.yml)
#     RUN   export HOME=/root PATH=/usr/local/bin:$PATH \
#     && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
#     && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
#     && /usr/local/bin/conda env create -f /app/docker/lasttwo.yml \
#     && /usr/local/bin/conda clean -y --all

#     # # Add modern C++ runtime to ehignlasttwo env to satisfy SciPy's GLIBCXX requirement
#     # RUN export HOME=/root PATH=/usr/local/bin:$PATH \
#     # && /usr/local/bin/conda install -n ehignlasttwo -c conda-forge -y libstdcxx-ng libgcc-ng \
#     # && /usr/local/bin/conda clean -y --all
#     # Install modern C++ runtime (optional but recommended to avoid GLIBCXX issues)
#     RUN export HOME=/root PATH=/usr/local/bin:$PATH \
#         && /usr/local/bin/conda install -n ehignlasttwo -c conda-forge -y libstdcxx-ng libgcc-ng \
#         && /usr/local/bin/conda clean -y --all

#     # Install PyTorch via conda and DGL from dglteam (use generic dgl package; CUDA compatibility via cudatoolkit)
#     RUN export HOME=/root PATH=/usr/local/bin:$PATH \
#         && /usr/local/bin/conda install -n ehignlasttwo -c pytorch -y pytorch=1.12.1 cudatoolkit=11.3 \
#         && /usr/local/bin/conda install -n ehignlasttwo -c dglteam -y dgl=0.9.1 \
#         && /usr/local/bin/conda install -n ehignlasttwo -c conda-forge -y pandas=2.2.3 \
#         && /usr/local/bin/conda install -n ehignlasttwo -c conda-forge -y rdkit=2024.09.4 \
#         && /usr/local/bin/conda clean -y --all

#  # Set working directory to /project
# WORKDIR "/app"

# syntax=docker/dockerfile:1.4

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set default RUN shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

ENV http_proxy="http://10.0.0.12:8001"
ENV https_proxy="http://10.0.0.12:8001"
ENV ftp_proxy="http://10.0.0.12:8001"

# Set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install basic packages for compiling and building
ENV DEBIAN_FRONTEND=noninteractive
# change apt source to Tsinghua mirror (optional)
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-7 \
    git \
    curl \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    libboost-all-dev \
    ibverbs-providers \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fontconfig \
    language-pack-en \
    sysstat \
    gnupg \
    lsb-release \
    sudo \
    && rm -rf /var/lib/apt/lists/*

    # Install Miniconda & use Python 3.9
ARG python=3.9
ENV PYTHON_VERSION=${python}
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install-conda.sh \
    && chmod +x /tmp/install-conda.sh \
    && bash /tmp/install-conda.sh -b -f -p /usr/local \
    && rm -f /tmp/install-conda.sh \
    && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda install -y python=${PYTHON_VERSION} \
    && conda clean -y --all

# # Setup TUNA mirror (optional)
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# RUN mkdir -p ~/.conda && \
#     echo "channels:" > ~/.condarc && \
#     echo "  - defaults" >> ~/.condarc && \
#     echo "show_channel_urls: true" >> ~/.condarc && \
#     echo "channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda" >> ~/.condarc && \
#     echo "default_channels:" >> ~/.condarc && \
#     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main" >> ~/.condarc && \
#     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free" >> ~/.condarc && \
#     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r" >> ~/.condarc && \
#     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro" >> ~/.condarc && \
#     echo "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2" >> ~/.condarc && \
#     echo "custom_channels:" >> ~/.condarc && \
#     echo "  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
#     echo "  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
#     echo "  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
#     echo "  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
#     echo "  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
#     echo "  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >> ~/.condarc && \
#     echo "  nvidia: https://mirrors.sustech.edu.cn/anaconda-extra/cloud" >> ~/.condarc

    # By default, install packages from `requirements.txt` with pip.
    COPY . /app

    # RUN   conda env create -f /app/docker/first.yml \
    # && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    # && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    # && conda clean -y --all

    # RUN   conda env create -f /app/docker/lasttwo.yml \
    # && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    # && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    # && conda clean -y --all

    #Ensure TOS accepted before creating environments (first.yml)
    RUN   export HOME=/root PATH=/usr/local/bin:$PATH \
    && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /usr/local/bin/conda env create -f /app/docker/first.yml \
    && /usr/local/bin/conda clean -y --all

    # # Ensure RDKit is available in ehignfirst environment (used by preprocessing)
    # RUN export HOME=/root PATH=/usr/local/bin:$PATH \
    #     && /usr/local/bin/conda install -n ehignfirst -c conda-forge -y rdkit=2024.09.4 \
    #     && /usr/local/bin/conda clean -y --all

    # Ensure TOS accepted before creating environments (lasttwo.yml)
    RUN   export HOME=/root PATH=/usr/local/bin:$PATH \
    && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /usr/local/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /usr/local/bin/conda env create -f /app/docker/lasttwo.yml \
    && /usr/local/bin/conda clean -y --all
 # Set working directory to /project
WORKDIR "/app"