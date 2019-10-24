from abc import abstractmethod

class Model:
    @abstractmethod
    def train_and_predict(self,train,vaild,test,param:dict):
        raise NotImplementedError
    
