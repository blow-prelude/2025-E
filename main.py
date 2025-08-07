# from moudles.serSystem import SerSystem
import myglobal
from moudles.videoSystem import VideoSystem


def main():
    myglobal.status = 1
    video_system = VideoSystem()
    # 与ti通信使用uart5
    # 与stm32通信使用uart4
    # ti_ser = SerSystem("/dev/AMA2")
    # 对对应uart3,gpio4:TXD3,gpio5:RXD3
    # stm_ser = SerSystem("/dev/AMA3")
    # 对应uart4,gpio8:TXD4,gpio9:RXD4
    while True:
        # sign_received = ti_ser.receiveSigns()
        sign_received = 1
        if sign_received in ("y2", "y3", "y4", "y5"):
            digit = int(sign_received[1])
            myglobal.status = digit - 1
        if sign_received == "n":
            myglobal.status = 0
        # 待机状态，等待信号
        if myglobal.status == 0:
            continue
        # 模式1，即第二问
        elif myglobal.status == 1:
            video_system.q2_process()
        # 模式2，即第三问
        elif myglobal.status == 2:
            video_system.q3_process()


if __name__ == "__main__":
    main()
