#ifndef TRAINING_OBSERVER_H
#define TRAINING_OBSERVER_H


template<typename T>
struct ITrainingObserver {
    virtual ~ITrainingObserver() = default;
    virtual void onEpochEnd(int epoch, int num_epochs, T loss) = 0;
};

template<typename T>
struct TrainingProgressDisplay: ITrainingObserver<T> {
    void onEpochEnd(int epoch, int num_epochs, T loss) override {
        int bar_width = 50; float ratio = float(epoch)/float(num_epochs);
        int pos = int(bar_width*ratio);
        std::cout << "[";
        for (int i=0; i<bar_width; ++i) {
            if (i < pos) std::cout << "="; else std::cout << " ";
        }
        std::cout << "] " << int(ratio*100.0f) << "%  " << "loss=" << loss << "\n";
    }
};


#endif //TRAINING_OBSERVER_H
