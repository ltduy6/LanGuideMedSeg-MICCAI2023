from engine.wrapper import LanGuideMedSegWrapper
from engine.wrapper_teacher import TeacherWrapper

def get_model(args):
    if args.model == "Teacher":
        return TeacherWrapper(args)
    elif args.model == "Student":
        return LanGuideMedSegWrapper(args)
    else:
        raise NotImplementedError('Model not implemented!')