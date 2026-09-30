from dynamixel_sdk import *

portHandler = PortHandler('/dev/ttyACM0')
packetHandler = PacketHandler(2.0)

portHandler.openPort()
portHandler.setBaudRate(1000000)  # 你使用的波特率

for i in range(1, 7):
    dxl_model_number, comm_result, error = packetHandler.ping(portHandler, i)
    if comm_result == COMM_SUCCESS:
        print(f"[OK] Motor ID {i} found, model: {dxl_model_number}")
    else:
        print(f"[FAIL] No response from ID {i}, error: {packetHandler.getTxRxResult(comm_result)}")
