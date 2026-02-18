FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    curl \
    git \
    wget \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    lsb-release \
    gnupg \
    software-properties-common

# ROS 2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN echo "deb http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list
RUN apt update && apt install -y ros-humble-desktop

# Gazebo Harmonic
RUN apt install -y gazebo

# PX4 dependencies
RUN pip3 install torch stable-baselines3 gym numpy matplotlib

WORKDIR /workspace

