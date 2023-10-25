import os
import torch.multiprocessing as mp

import torch
import torch.distributed as dist

from jjuke import logger, options
from jjuke.net_utils import seed_everything, find_free_port
from jjuke.net_utils.dist import safe_barrier


def main_worker(rank, args):
    if args.ddp:
        dist.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=rank)

    args.rank = rank
    args.rankzero = rank == 0
    args.gpu = args.gpus[rank]
    torch.cuda.set_device(args.gpu)

    if args.rankzero:
        logger.basic_config(args.exp_path / "main.log")
    else:
        logger.basic_config(None, lock=True)
    args.log = logger.get_logger()

    args.seed += rank
    seed_everything(args.seed)

    if args.ddp:
        print("main_worker with rank:{} (gpu:{}) is loaded".format(rank, args.gpu))
    else:
        print("main_worker with gpu:{} in main thread is loaded".format(args.gpu))

    trainer = options.instantiate_from_config(args.trainer, args)
    trainer.fit()

    safe_barrier()


def main():
    args = options.get_config()

    os.environ["OMP_NUM_THREADS"] = str(min(args.dataset.params.num_workers, mp.cpu_count()))

    args.world_size = len(args.gpus)
    args.ddp = args.world_size > 1
    port = find_free_port()
    args.dist_url = "tcp://127.0.0.1:{}".format(port)

    if args.ddp:
        pc = mp.spawn(main_worker, nprocs=args.world_size, args=(args,), join=False)
        pids = " ".join(map(str, pc.pids()))
        print("\33[101mProcess Ids:", pids, "\33[0m")
        try:
            pc.join()
        except KeyboardInterrupt:
            print("\33[101mkill {:s}\33[0m".format(pids))
            os.system("kill {:s}".format(pids))
    else:
        main_worker(0, args)


if __name__ == "__main__":
    main()
