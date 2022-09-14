FROM ubuntu:18.04

RUN apt update \
    &&  DEBIAN_FRONTEND=noninteractive apt install -y \
        build-essential \
        cmake \
        libopencv-dev

WORKDIR /app
COPY . .
RUN mkdir build && cd build && cmake .. && make

WORKDIR /app/build
RUN ./main_train

ENTRYPOINT ["./main_test"]