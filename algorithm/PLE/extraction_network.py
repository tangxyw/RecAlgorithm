import tensorflow as tf


def extraction_network(input, task_names, num_experts_per_task, num_experts_in_shared, expert_hidden_units, name):
    """
        实现PLE模型中的extraction_network模块
    Args:
        input (tf.Tensor):  输入, shape=(B, input_dim)
        task_names (list): 每个task的名字组合的list
        num_experts_per_task (list): 每个任务的task专家个数
        num_experts_in_shared (list): shared专家个数
        expert_hidden_units (int): 专家维度
        name (str): 传入tf.variable_scope的参数

    Returns:
        tf.Tensor, 输出, shape=(B, output_dim)
    """

    # 最终输出
    final_output = []

    # 所有task专家和shared专家
    all_experts = []

    with tf.variable_scope(name):
        # shared专家网络输出列表
        shared_specific_experts = [tf.layers.dense(input,
                                                   expert_hidden_units,
                                                   activation=tf.nn.relu,
                                                   name=f"shared_expert_{i}") for i in range(num_experts_in_shared)]
        # (B, expert_hidden_units) * num_experts_in_shared

        shared_specific_experts = [e[:, tf.newaxis, :] for e in shared_specific_experts]    # (B, 1, expert_hidden_units) * num_experts_in_shared
        shared_specific_experts = tf.concat(shared_specific_experts, axis=1)    # (B, num_experts_in_shared, expert_hidden_units)

        # 遍历每个任务的task专家数量, 与share专家组合
        for task_name, num_experts_in_task in zip(task_names, num_experts_per_task):
            # 每个任务的task专家网络输出列表
            task_specific_experts = [tf.layers.dense(input,
                                                     expert_hidden_units,
                                                     activation=tf.nn.relu,
                                                     name=f"task_specific_expert_{task_name}_{i}") for i in
                                                          range(num_experts_in_task)]
            # (B, expert_hidden_units) * num_experts_in_task

            task_specific_experts = [e[:, tf.newaxis, :] for e in task_specific_experts]  # (B, 1, expert_hidden_units) * num_experts_in_task
            task_specific_experts = tf.concat(task_specific_experts, axis=1)  # (B, num_experts_in_task, expert_hidden_units)
            # 放入all_experts内, 后面用
            all_experts.append(task_specific_experts)

            combined_experts = tf.concat([task_specific_experts, shared_specific_experts], axis=1)
            # (B, num_experts_in_task+num_experts_in_shared, expert_hidden_units)

            # 门输出
            gate = tf.layers.dense(input,
                                   num_experts_in_task+num_experts_in_shared,
                                   activation=tf.nn.softmax,
                                   use_bias=False,  # 论文中省略了bias
                                   name=f"gate_{task_name}")
            # (B, num_experts_in_task+num_experts_in_shared)

            gate = tf.expand_dims(gate, axis=-1)  # (B, num_experts_in_task+num_experts_in_shared, 1)
            # 组合专家网络输出和每一个任务相关的门输出做矩阵乘法
            task_output = tf.matmul(combined_experts, gate, transpose_a=True)  # (B, expert_hidden_units, 1)
            task_output = tf.squeeze(task_output, axis=-1)  # (B, expert_hidden_units)
            final_output.append(task_output)

        # 所有专家组合
        all_experts.append(shared_specific_experts)
        all_experts = tf.concat(all_experts, axis=1)    # (B, sum(num_experts_per_task)+num_experts_in_shared, expert_hidden_units)
        # 所有专家对应的门输出
        all_gate = tf.layers.dense(input,
                                   sum(num_experts_per_task)+num_experts_in_shared,
                                   activation=tf.nn.softmax,
                                   use_bias=False,  # 论文中省略了bias
                                   name=f"all_gate")
        # (B, sum(num_experts_per_task)+num_experts_in_shared)
        all_gate = tf.expand_dims(all_gate, axis=-1)    # (B, sum(num_experts_per_task)+num_experts_in_shared), 1)
        # 所有专家组合网络输出和对应门输出做矩阵乘法
        all_output = tf.matmul(all_experts, all_gate, transpose_a=True)  # (B, expert_hidden_units, 1)
        all_output = tf.squeeze(all_output, axis=-1)    # (B, expert_hidden_units)
        # 添加到最终输出
        final_output.append(all_output)

    return tf.add_n(final_output)   # (B, expert_hidden_units)