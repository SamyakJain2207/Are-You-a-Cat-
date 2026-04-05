from abc import ABC, abstractmethod

class Stage(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def requires(self):
        '''
        return list of required artifacts
        '''
        pass

    @abstractmethod
    def produces(self):
        '''
        returns list of produced artifacts
        '''
        pass

    @abstractmethod
    def is_complete(self):
        '''
        return True if the Stage is complete
        '''
        pass

    @abstractmethod
    def run(self):
        '''
        runs the transformation
        '''
        pass

    @abstractmethod
    def validate(self):
        '''
        verify the output is correct
        '''
        pass

class PipelineRunner:
    def __init__(self, stages):
        self.stages = stages

    def run(self):
        for stage in self.stages:
            if stage.is_complete():
                print(f"Skipping stage: {stage.name} as it is already completed")
                continue
            print(f"Running stage: {stage.name}")
            stage.run()
            stage.validate()