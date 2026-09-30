python3 lerobot/scripts/server/robot_client.py \
    --server_address=127.0.0.1:8080 \
    --robot.type=send_robot \
    --robot.id=my_rm_robot \
    --task="Grasp a lego block and put it in the bin." \
    --robot.cameras '{
        front: {
            type: opencv,
            index_or_path: 0,
            width: 1280,
            height: 720,
            fps: 30
        }
        }'  \
    --robot.teleop=false \
    --fps=30 \
    --policy_type=smolvla \
    --pretrained_name_or_path=/home/sdznkj/zkx_ws/sendlerobot/outputs/test/060000/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=false
# Optional linear interp (default 1=off): --interpolation_multiplier=2