- The TCP folder contains cross-language distributed reinforcement learning written using the TCP communication protocol
    - ppo1_async file define two network implementations to handle the old and new policies
    - client file represent the original language, such as C++; server file represent using python to connect the algorithm

- ppo2_async file use actor network represent the old policy, and dont't use smaple pool and model pool to update leaner policy

- ppo3_async file use actor network represent the old policy, and use smaple pool and model pool to update leaner policy

