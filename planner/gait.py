from enum import Enum
import numpy as np

# Reference: https://github.com/yinghansun/pympc-quadruped/

class Gait(Enum):
    '''
    name: name of the gait
    num_segment: number of segments in one period
    
    stance_offsets: stance offset for each leg, offset from which leg starts to 
                    stance/stand on the ground
    stance_durations: stance duration for each leg
    '''

    STANDING = 'standing', 16, np.array([0, 0, 0, 0]), np.array([16, 16, 16, 16])
    TROTTING160 = 'trotting', 160, np.array([0, 80, 80, 0]), np.array([80, 80, 80, 80])
    TROTTING10 = 'trotting', 10, np.array([0, 5, 5, 0]), np.array([5, 5, 5, 5])
    JUMPING16 = 'jumping', 16, np.array([0, 0, 0, 0]), np.array([4, 4, 4, 4])
    PACING16 = 'pacing', 16, np.array([8, 0, 8, 0]), np.array([8, 8, 8, 8])
    PACING10 = 'pacing', 10, np.array([5, 0, 5, 0]), np.array([5, 5, 5, 5])

    def __init__(
        self, 
        name: str, 
        num_segment: int, 
        stance_offsets: np.ndarray, 
        stance_durations: np.ndarray
    ) -> None:

        self.num_segment = num_segment
        self.stance_offsets = stance_offsets
        self.stance_durations = stance_durations

        self.dt_control: float = 8e-3

        # each leg has the same swing time and stance time in one period
        self.total_swing_time: int = num_segment - stance_durations[0]
        self.total_stance_time: int = stance_durations[0]

        # normalization
        self.stance_offsets_normalized = stance_offsets / num_segment
        self.stance_durations_normalized = stance_durations / num_segment


    @property
    def swing_time(self) -> float:
        '''
        compute total swing time
        return total swing time in one period = dt_control * total_swing_time
        '''
        return self.dt_control * self.total_swing_time

    @property
    def stance_time(self) -> float:
        '''
        compute total stance time
        return total stance time in one period = dt_control * total_stance_time
        '''
        return self.dt_control * self.total_stance_time


    def set_iteration(self, cur_iteration: int) -> None:
        self.iteration = cur_iteration % self.num_segment
        self.phase = (cur_iteration % self.num_segment) / (self.num_segment)


    def get_gait_table(self):
        '''
        compute gait table for force constraints in mpc

        True if the leg is in stance phase
        False if the leg is in swing phase
        '''
        gait_table = [False, False, False, False]
        segment_idx = (self.iteration) % self.num_segment

        # Each element in cur_segment represents the distance of the current segment from the stance offset
        # If the distance is more than the stance duration, the leg is in swing phase
        cur_segment = segment_idx - self.stance_offsets
        for j in range(4):
            if cur_segment[j] < 0:
                cur_segment[j] += self.num_segment
            
            if cur_segment[j] < self.stance_durations[j]:
                gait_table[j] = True
            else:
                gait_table[j] = False

        return gait_table

    def get_swing_state(self) -> np.ndarray:
        swing_offsets_normalizerd = self.stance_offsets_normalized + self.stance_durations_normalized
        swing_durations_normalized = 1 - self.stance_durations_normalized

        phase_state = np.array([self.phase, self.phase, self.phase, self.phase], dtype=np.float32)
        swing_state = phase_state - swing_offsets_normalizerd

        for i in range(4):
            if swing_state[i] < 0:
                swing_state[i] += 1
            
            if swing_state[i] > swing_durations_normalized[i]:
                swing_state[i] = 0
            else:
                swing_state[i] = swing_state[i] / swing_durations_normalized[i]

        return swing_state

    def get_stance_state(self) -> np.ndarray:
        phase_state = np.array([self.phase, self.phase, self.phase, self.phase], dtype=np.float32)
        stance_state = phase_state - self.stance_offsets_normalized
        for i in range(4):
            if stance_state[i] < 0:
                stance_state[i] += 1
            
            if stance_state[i] > self.stance_durations_normalized[i]:
                stance_state[i] = 0
            else:
                stance_state[i] = stance_state[i] / self.stance_durations_normalized[i]

        return stance_state


def main():
    trotting = Gait.TROTTING10

    idx = 0

    print(trotting.swing_time)

    for i in range(20):
        print('--------update MPC--------')
        print('idx =', idx)
        trotting.set_iteration(i)
        idx += 1
        print(trotting.get_gait_table())

            
        # print(trotting.get_cur_swing_time(30 * 0.001))
        # print(trotting.swing_time)

        stance_state = trotting.get_stance_state()
        swing_state = trotting.get_swing_state()
        # print('iteration: ', trotting.iteration)

        # print('phase: ', trotting.phase)
        print('stance state: ', stance_state)
        print('swing state: ', swing_state)



if __name__ == '__main__':
    main()

