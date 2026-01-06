from keras import ops


class FrameBuffer:
    def __init__(self, image_shape, batch_size=None, buffer_size=2):
        """
        image_shape: [h, w, c]
        buffer_size: number of temporal frames stored
        """
        self.image_shape = tuple(image_shape)
        self.buffer_size = buffer_size
        sample_shape = image_shape[:-1] if batch_size is None else [batch_size, *image_shape[:-1]]
        self.buffer = ops.zeros(
            [*sample_shape, buffer_size]
        )  # shape: [h, w, buffer_size]

    def __getitem__(self, key):
        return self.buffer[key]

    def __setitem__(self, key, value):
        self.buffer = ops.slice_update(self.buffer, key, value)
        return self.buffer[key]

    def shift(self, new_item):
        self.buffer = lifo_shift(self.buffer, new_item)  # assumed defined elsewhere

    def latest(self):
        return self.buffer[..., -1]

    def is_populated(self):
        return not ops.all(self.buffer[..., -2] == 0) and not ops.all(
            self.buffer[..., -1] == 0
        )

def lifo_shift(existing_buffer, new_item):
    return ops.concatenate([existing_buffer[..., 1:], new_item], axis=-1)
