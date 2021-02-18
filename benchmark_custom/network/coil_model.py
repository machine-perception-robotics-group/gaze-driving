"""
    A simple factory module that returns instances of possible modules 

"""

from .models import CoILICRA
from .models import CoILGaze
from .models import CoABNDrive
from .models import NormalDrive
from .models import CoABNDriveGaze


def CoILModel(architecture_name, architecture_configuration):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if architecture_name == 'coil-icra':

        return CoILICRA(architecture_configuration)

    elif architecture_name == 'coil-gaze':

        return CoILGaze(architecture_configuration)

    elif architecture_name == 'coil-gaze-loss':

        return CoILICRA(architecture_configuration)

    elif architecture_name == 'abn-drive':
        ## 視線マスク有りABN
        if 'gmask' in architecture_configuration:
            ## todo: 視線マスクとその他のABNをプログラムを分けずに統合したい
            return CoABNDriveGaze(architecture_configuration)

        ## 通常のABN (or Loss version)    
        return CoABNDrive(architecture_configuration)

    elif architecture_name == 'normal-drive':
        return NormalDrive(architecture_configuration)

    else:

        raise ValueError(" Not found architecture name")