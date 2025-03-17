import tensorflow as tf
from sumo_rl.environment.env import SumoEnvironment

# Load mô hình Keras từ file .h5
trained_model = tf.keras.models.load_model("trained_kernel_initialiser_reward.h5")

# Tích hợp mô hình vào môi trường SUMO
env = SumoEnvironment(net_file="cross.net.xml",
                      route_file="cross.rou.xml",
                      use_gui=True)

obs = env.reset()
done = False

while not done:
    action = trained_model.predict(obs)  # Đầu ra có thể cần chuyển đổi
    obs, reward, done, _ = env.step(action)
