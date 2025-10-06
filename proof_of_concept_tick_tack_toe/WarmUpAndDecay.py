import tensorflow as tf

class WarmUpAndDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, decay_steps):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        decay_lr = self.base_lr * tf.math.exp(-0.1 * (step - self.warmup_steps) / self.decay_steps)
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decay_lr)
    
    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
        }