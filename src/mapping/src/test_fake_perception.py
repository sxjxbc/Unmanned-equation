#!/usr/bin/env python3
print("=== 调试版本启动 ===")

import sys
import traceback

try:
    print("1. 导入 rospy...")
    import rospy
    print("   ✓ rospy 导入成功")
    
    print("2. 导入 numpy...")
    import numpy as np
    print("   ✓ numpy 导入成功")
    
    print("3. 导入消息类型...")
    from sensor_msgs.msg import NavSatFix, Imu
    print("   ✓ sensor_msgs 导入成功")
    
    try:
        from mapping.msg import ConeDetection
        print("   ✓ mapping/ConeDetection 导入成功")
    except ImportError as e:
        print(f"   ✗ mapping/ConeDetection 导入失败: {e}")
        print("   这可能是因为消息没有正确生成")
        sys.exit(1)
    
    print("4. 初始化 ROS 节点...")
    rospy.init_node('fsae_simulator_test', anonymous=False)
    print("   ✓ 节点初始化成功")
    
    print("5. 创建发布器...")
    gps_pub = rospy.Publisher('/GPS_data_test', NavSatFix, queue_size=10)
    imu_pub = rospy.Publisher('/imu_data_test', Imu, queue_size=10)
    cone_pub = rospy.Publisher('/perception/cones_test', ConeDetection, queue_size=10)
    print("   ✓ 发布器创建成功")
    
    print("🎉 所有测试通过！脚本应该可以正常运行")
    print("按 Ctrl+C 退出")
    
    # 保持运行
    rospy.spin()
    
except Exception as e:
    print(f"✗ 错误: {e}")
    traceback.print_exc()
