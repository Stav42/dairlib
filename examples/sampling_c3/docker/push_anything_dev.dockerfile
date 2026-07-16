# Build from examples/sampling_c3/docker (snopt/gurobi tarballs must be in this directory):
#   docker build -f push_anything_dev.dockerfile -t push-anything-env .
FROM ubuntu:noble
ARG DRAKE_VERSION=1.51.1
ARG DEBIAN_FRONTEND=noninteractive

# Some networks block plain HTTP (port 80) to Ubuntu's mirrors while HTTPS
# (443) works fine — force apt to use https:// for all repositories, both
# the legacy sources.list format and noble's new deb822 *.sources format.
RUN for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do \
        [ -f "$f" ] && sed -i 's|http://|https://|g' "$f"; \
    done; true

# Bootstrap chicken-and-egg: this base image has no CA trust store yet, so
# apt can't verify the https:// mirror's certificate (the cert itself is
# fine — there's just nothing here to check it against, since
# ca-certificates is one of the packages we're about to install). Disable
# verification for JUST this one call; once ca-certificates lands, every
# later RUN apt-get in this Dockerfile verifies normally.
RUN apt-get -o Acquire::https::Verify-Peer=false update && \
    apt-get -o Acquire::https::Verify-Peer=false install -y \
    wget lsb-release pkg-config zip g++ zlib1g-dev unzip ca-certificates gnupg git \
    libopenblas-dev openjdk-17-jdk \
    iproute2 gosu \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/RobotLocomotion/drake/archive/v${DRAKE_VERSION}.tar.gz \
    && tar -xzf v${DRAKE_VERSION}.tar.gz drake-${DRAKE_VERSION}/setup/ \
    && ./drake-${DRAKE_VERSION}/setup/install_prereqs --developer -y --with-bazel --with-clang \
    && rm -rf v${DRAKE_VERSION}.tar.gz drake-${DRAKE_VERSION}/

# Procman (libbot2 not in Drake apt for noble; build procman only from source).
RUN apt-get update && apt-get install -y \
    cmake build-essential liblcm-dev python3-lcm python3-pip python3-setuptools \
    libglu1-mesa-dev libgtk-3-dev \
    && git clone --depth 1 --branch drake https://github.com/RobotLocomotion/libbot2.git /tmp/libbot2 \
    && cmake -S /tmp/libbot2 -B /tmp/libbot2-build -DCMAKE_INSTALL_PREFIX=/opt/libbot2 \
    && cmake --build /tmp/libbot2-build -j"$(nproc)" --target bot2-procman \
    && cmake --install /tmp/libbot2-build \
    && rm -rf /tmp/libbot2 /tmp/libbot2-build /var/lib/apt/lists/*
ENV PATH="/opt/libbot2/bin:${PATH}"

COPY snopt7.6.tar.gz /snopt7.6.tar.gz
ENV SNOPT_PATH=/snopt7.6.tar.gz

COPY gurobi10.0.3_linux64.tar.gz /tmp/gurobi.tar.gz
RUN mkdir -p /opt/gurobi && tar xzf /tmp/gurobi.tar.gz -C /opt/gurobi/ && rm /tmp/gurobi.tar.gz
ENV GUROBI_HOME=/opt/gurobi/gurobi1003/linux64
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Same packages as examples/sampling_c3/sampling_generation/python_requirements.txt
RUN pip3 install --break-system-packages \
    trimesh ruamel.yaml lxml fast_simplification pyglet vhacdx

# install_prereqs creates ubuntu (uid 1000)
RUN usermod -l pushanything -d /home/pushanything -m ubuntu

WORKDIR /home/pushanything

# No `USER pushanything` here (deliberately) — the container must start as
# root so entrypoint.sh can do privileged network setup (enable loopback
# multicast, required by LCM) before dropping to the unprivileged user
# itself via gosu. Requires `docker run --cap-add=NET_ADMIN`; without it,
# entrypoint.sh logs a warning and continues (you get a working shell, LCM
# just won't work until you add the flag and recreate the container).
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
