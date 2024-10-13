import pynvml
import time


def sleep_until_gpu_free(free_memory_need=None, sleep_interval=180):
    """

    :param free_memory_need:  a list to mark you want to used the number of gpu, if value is [0, 1], then mean using value * gpu cap
    :param sleep_interval: where gpu is busy, how many second do you want to detection again
    :return: None
    """
    pynvml.nvmlInit()
    gpus = pynvml.nvmlDeviceGetCount()
    print('we have {} gpu(s)'.format(gpus))
    handle = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in range(gpus)]
    memory_info = [pynvml.nvmlDeviceGetMemoryInfo(handles) for handles in handle]
    if free_memory_need is None:
        free_memory_need = [0.9 * memory_info[idx].total for idx in range(gpus)]
    else:
        if isinstance(free_memory_need, list):
            if len(free_memory_need) == 0:
                return
            for i in range(min(len(free_memory_need), gpus)):
                if free_memory_need[i] <= 1.0:
                    free_memory_need[i] = free_memory_need[i] * memory_info[i].total
        else:
            free_memory_need = [0.9 * memory_info[idx].total for idx in range(gpus)]
    busy = True
    try_times = 0
    while busy:
        busy = False
        try_times += 1
        print("we {} times to try to load gpu information".format(try_times))
        handle = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in range(gpus)]
        memory_info = [pynvml.nvmlDeviceGetMemoryInfo(handles) for handles in handle]
        for idx in range(min(gpus, len(free_memory_need))):
            if memory_info[idx].free < free_memory_need[idx]:
                print('gpu{} busy: used:{}/total:{}'.format(idx, memory_info[idx].used, memory_info[idx].total))
                busy = True
        if busy:
            print("gpu(s) is busy, sleep\n")
            time.sleep(sleep_interval)
    print("gpus is free, runing!")


if __name__ == '__main__':
    sleep_until_gpu_free([0,0,0.9,0.9])
    print('done')

