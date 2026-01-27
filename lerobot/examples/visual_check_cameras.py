import cv2
import time

def show_all_cameras_with_details(max_check=10):
    """
    功能：
    1. 扫描摄像头，并在控制台打印详细的硬件参数（ID, API, 分辨率, FPS等）。
    2. 扫描结束后，同时打开所有可用的摄像头窗口进行实时预览。
    """
    print(f"正在扫描前 {max_check} 个接口...\n")
    print("--- Detected Cameras ---")
    
    # 1. 列表初始化：用来存放所有“健康”的摄像头对象和ID
    valid_caps = []       
    valid_indices = []    
    
    for index in range(max_check):
        # 尝试打开摄像头
        cap = cv2.VideoCapture(index)
        
        if cap.isOpened():
            # --- [新增功能] 获取详细硬件信息 ---
            # 获取后端 API 名称 (如 MSMF, DSHOW, V4L2)
            backend_name = cap.getBackendName()
            
            # 获取分辨率和帧率
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fmt = cap.get(cv2.CAP_PROP_FORMAT)
            
            # 构造显示的名称
            camera_name = f"OpenCV Camera @ {index}"
            
            # --- [新增功能] 打印详细信息块 ---
            print(f"Camera #{index}:")
            print(f"  Name: {camera_name}")
            print(f"  Type: OpenCV")
            print(f"  Id: {index}")
            print(f"  Backend api: {backend_name}")
            print(f"  Default stream profile:")
            print(f"    Format: {fmt}")
            print(f"    Width: {int(width)}")
            print(f"    Height: {int(height)}")
            print(f"    Fps: {fps}")
            print("-" * 20)
            
            # --- 验证摄像头是否能真正读取画面 ---
            # 有些摄像头能打开（isOpened为True）但读不出数据（黑屏或报错）
            ret, frame = cap.read()
            
            if ret:
                # 如果能读到画面，说明完全可用
                # 将其加入列表，稍后用于显示
                valid_caps.append(cap)
                valid_indices.append(index)
            else:
                # 如果读不到数据，打印警告并释放资源
                print(f"[警告] Camera #{index} 无法读取画面 (No Frame)，已忽略。")
                print("-" * 20)
                cap.release()
        else:
            # 无法打开的索引直接跳过，不打印任何信息
            pass

    # 如果没有找到任何摄像头
    if not valid_caps:
        print("\n[结果] 未找到任何可用摄像头。程序结束。")
        return

    # --- 进入实时显示阶段 ---
    print(f"\n[就绪] 共检测到 {len(valid_caps)} 个可用摄像头。")
    print("正在启动多窗口显示...")
    print("操作说明: 按键盘上的【q】键可同时关闭所有窗口并退出程序。")

    while True:
        # 标记位：假设所有摄像头都在正常工作
        all_working = True
        
        # 遍历所有已保存的摄像头对象
        for i, cap in enumerate(valid_caps):
            camera_id = valid_indices[i]
            
            # 读取一帧
            ret, frame = cap.read()
            
            if ret:
                # 在画面左上角写上 ID，方便肉眼识别
                text = f"Cam ID: {camera_id} | {int(cap.get(3))}x{int(cap.get(4))}"
                cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)
                
                # 定义窗口名字
                window_name = f"Camera View - ID {camera_id}"
                
                # 显示画面
                cv2.imshow(window_name, frame)
                
                # 简单的窗口排版逻辑
                # 第1个窗口在 (50, 100)，第2个在 (450, 100)...
                if i < 4: 
                    cv2.moveWindow(window_name, 50 + (i * 400), 100)
            else:
                print(f"[错误] 摄像头 {camera_id} 信号丢失。")
                all_working = False

        # 监听键盘按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户请求退出。")
            break
        
        # 如果有设备断开，退出循环
        if not all_working:
            print("因设备断开，程序结束。")
            break

    # --- 清理工作 ---
    print("正在释放资源...")
    for cap in valid_caps:
        cap.release()
    cv2.destroyAllWindows()
    print("已退出。")

if __name__ == "__main__":
    show_all_cameras_with_details()