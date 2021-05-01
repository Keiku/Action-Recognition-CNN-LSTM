FROM nvidia/cuda:11.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# install some basic utilities
RUN apt-get update && apt-get install -y \
  curl \
  ca-certificates \
  sudo \
  git \
  tzdata \
  ffmpeg

ARG TIMEZONE
RUN ln -sf /usr/share/zoneinfo/${TIMEZONE} /etc/localtime

# install python
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  pipenv \
  && rm -rf /var/lib/apt/lists/*

RUN ln -s /bin/python3.8 /bin/python
RUN ln -s /usr/bin/pip3.8 /usr/bin/pip
RUN pip3 install -U pip

# set the user created by entrypoint to sudouser
ENV USER_NAME=docker
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER_NAME}
RUN chmod u+s /usr/sbin/useradd \
  && chmod u+s /usr/sbin/groupadd

# install python virtualenv from Pipfile
COPY Pipfile ./
COPY Pipfile.lock ./
RUN pipenv install --system

# create a user with the same id as docker host
COPY docker/entrypoint.sh /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]

# set the default command
CMD ["bash"]
