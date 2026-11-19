from enum import Enum

class Fid01Mode(Enum):
    OVER_ALIGNED = auto() # +k whole blocks
    UNDER_ALIGNED = auto() # -k whole blocks
    MISALIGNED = auto() # +/- partial block
    
def inject_fid01(resume_point_index, mode=fid01_mode.OVERCALCULATE.value):
    if mode == 0:
        return overcalculate(resume_point_index)
    elif mode == 1:
        return mode01_undercalculate(resume_point_index)
    elif mode == 2:
        return mode02_misalignment(resume_point_index)
        
def mode00_overcalculate(resume_point_index, blocksize):
    return (resume_point_index + blocksize // 2) % blocksize

def mode01_undercalculate(resume_point_index, blocksize):
    return (resume_point_index - blocksize // 2) % blocksize

def mode02_misalignment(resume_point_index, block):
    # TODO
    return # beginning of block 0