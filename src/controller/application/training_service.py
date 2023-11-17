from ..domain.robotica import Coppelia, P3DX

class TrainingService:

    @staticmethod
    def run():
        coppelia = Coppelia()
        robot = P3DX(coppelia.sim, 'PioneerP3DX')
        robot.set_speed(+1.2, -1.2)
        coppelia.start_simulation()
        while (t := coppelia.sim.getSimulationTime()) < 3:
            print(f'Simulation time: {t:.3f} [s]')
        coppelia.stop_simulation()


if __name__ == '__main__':
    TrainingService.run()
    