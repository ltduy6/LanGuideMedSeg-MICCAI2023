from engine.wrapper import LanGuideMedSegWrapper
from engine.wrapper_teacher import TeacherWrapper
from engine.wrapper_kd_baseline import BaselineKDWrapper

def get_model(args):
    if args.model == "Teacher":
        return TeacherWrapper(args)
    elif args.model == "Student":
        return LanGuideMedSegWrapper(args)
    elif args.model == "BaselineKD":
        return BaselineKDWrapper(args)
    else:
        raise NotImplementedError('Model not implemented!')