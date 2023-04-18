import os
import time
import argparse
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from scipy.ndimage.filters import gaussian_filter1d
from scipy.special import logit

from load_data import LoadData, setup_seed
from cnn_model import ConvNN
from active_learning import select_acq_function, active_learning_procedure







def load_CNN_model(args, device):
    """Load new model each time for different acqusition function
    each experiments"""
    model = ConvNN().to(device)
    cnn_classifier = NeuralNetClassifier(
        module=model,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device
    )
    return cnn_classifier


def save_as_npy(data: np.ndarray, folder: str, name: str):
    """Save result as npy file

    Attributes:
        data: np array to be saved as npy file,
        folder: result folder name,
        name: npy filename
    """
    file_name = os.path.join(folder, name + ".npy")
    np.save(file_name, data)
    print(f"Saved: {file_name}")


def plot_results(data: dict):
    """Plot results histogram using matplotlib"""
    sns.set()
    for key in data.keys():
        # data[key] = gaussian_filter1d(data[key], sigma=0.9) # for smoother graph
        plt.plot(data[key], label=key.split('-')[0])
    plt.legend()
    plt.show()


def print_elapsed_time(start_time: float, exp: int, acq_func: str):
    """Print elapsed time for each experiment of acquiring

    Attributes:
        start_time: Starting time (in time.time()),
        exp: Experiment iteration
        acq_func: Name of acquisition function
    """
    elp = time.time() - start_time
    print(
        f"********** Experiment {exp} ({acq_func}): {int(elp//3600)}:{int(elp%3600//60)}:{int(elp%60)} **********"
    )

def experiment_iter(args, device, datasets, acq_func,state, beta,ratio,seeds,totest):
    """
    Conduct experiment to get accuracy curve and val/test accuracy
    Attributes:
        args: Argparse input,
        device: Cpu or gpu,
        datasets: Dataset dict that consists of all datasets,
        acq_func: query strategy
        beta: weight parameter for weighted product query
        state: whether to use dropout
        totest: choose validation or test set to test the model
    """
    avg_hist=[]
    scores = []
    """
    start the experiment, train the model and eavluate on the target set, record accuracy curve and val/test accuracy
    """
    for e in range(args.experiments):
        setup_seed(int(seeds[e]))
        start_time = time.time()
        estimator = load_CNN_model(args, device)

        
        print(
            f"********** Experiment Iterations: {e+1}/{args.experiments} **********"
        )
        
        training_hist, val_score = active_learning_procedure(
            query_strategy=acq_func,
            X_val=datasets["X_val"],
            y_val=datasets["y_val"],
            X_test=datasets["X_test"],
            y_test=datasets["y_test"],
            X_pool=datasets["X_pool"],
            y_pool=datasets["y_pool"],
            X_init=datasets["X_init"],
            y_init=datasets["y_init"],
            estimator=estimator,
            training=state,
            args=args,
            beta=beta,
            ratio=ratio,
            totest=totest
        )
        avg_hist.append(training_hist)
        scores.append(val_score)
        print_elapsed_time(start_time, e + 1, acq_func)

    avg_score=sum(scores) / len(scores)
    """
    If testing on validation set, only return val accuracy, otherwise return accuracy curve and test accuracy
    """
    if totest == False:
        print('current accuracy:' + str(avg_score))
        return avg_score
    else:
        print('final accuracy:'+ str(avg_score))
        avg_hist = np.average(np.array(avg_hist), axis=0)
        return avg_hist, avg_score


def train_active_learning(args, device, datasets: dict) -> dict:
    """Start training process

    Attributes:
        args: Argparse input,
        estimator: Loaded model, e.g. CNN classifier,
        device: Cpu or gpu,
        datasets: Dataset dict that consists of all datasets,
    """
    acq_functions = select_acq_function((args.uncertainty,args.diversity))
    seeds=np.random.choice(range(10000),size=args.experiments,replace=False)
    
    results = dict()
    result_para = dict()
    if args.determ:
        state_loop = [True, False]  # dropout VS non-dropout
    else:
        state_loop = [True]  # run dropout only

    """
    Choose the query method
    """
    if args.runmode == 0:
        print("This is weighted query")
        if args.time_decay == True:
            print("This is time decay version")
        else:
            print("This is constant weight version")

        if args.sum_product == 1:
            print("This is weighted product")
        else:
            print("This is weighted sum")

        """
        If using weighted product query and no beta specified, conduct hyperparameter tuning
        """
        if args.beta==100:
            print("No beta specified, conduct hyperparameter tuning")
            for state in state_loop:
                for i, acq_func in enumerate(acq_functions):
                    acq_func_name = str(acq_func).split(" ")[1] + "-MC=" + str(state)

                    if args.time_decay==True:
                        acq_func_name+="time_decay"

                    if args.sum_product == 1:
                        acq_func_name+="product"
                    else:
                        acq_func_name+="sum"


                    if str(acq_func).split(" ")[1] == "uniform":
                        avg_hist, avg_test =experiment_iter(args=args, device=device, datasets=datasets, acq_func=acq_func,state=state, beta=args.beta,
                            ratio=args.candidate_ratio,seeds=seeds,totest=True)
                        results[acq_func_name] = avg_hist
                        result_para[acq_func_name]=np.array([avg_test])
                        break 


                    """
                    To conduct hyperparameter tuning using Gaussian process, get 3 initial observations
                    """
                    beta_test=np.linspace(0.1, 0.9, 9)
                    """
                    Since original beta is in [0, 1], which maybe too narrow. Transform it using logit function
                    """
                    
                    beta_test_value=[]

                    for b in beta_test:
                        print('current beta: '+str(b))

                        avg_val=experiment_iter(args=args, device=device, datasets=datasets, acq_func=acq_func,state=state, beta=b,
                            ratio=args.candidate_ratio,seeds=seeds, totest=False)

                        beta_test_value.append(avg_val)

                    
                    """
                    Find final best beta. Test the model on the test set.
                    """
                    idx=np.argsort(-np.array(beta_test_value))[0]
                    beta_final=beta_test[idx]
                    print('final beta:'+str(beta_final))

                    """
                    avg_hist records the accuracy curve on query times
                    """
                    avg_hist, avg_test =experiment_iter(args=args, device=device, datasets=datasets, acq_func=acq_func,state=state, beta=beta_final,
                        ratio=args.candidate_ratio,seeds=seeds,totest=True)

                    """
                    Save the accuracy curve across query times, best beta and test accuracy
                    """
                    results[acq_func_name] = avg_hist
                    result_para[acq_func_name]=np.array([beta_final, avg_test])
            """
            save results
            """
            save_as_npy(data=results, folder=args.result_dir, name=acq_func_name)
            save_as_npy(data=result_para,folder=args.result_dir,name=acq_func_name+"para")

        else:
            print("beta is specified, directly run the model")
            for state in state_loop:
                for i, acq_func in enumerate(acq_functions):
        
                    if args.time_decay==False:
                        acq_func_name = str(acq_func).split(" ")[1] + "-MC=" + str(state)+"-betagiven"+str(args.beta)
                    else:
                        acq_func_name = str(acq_func).split(" ")[1] + "-MC=" + str(state) + "-betagiven"+str(args.beta)+"-decay"

                    if args.sum_product == 1:
                        acq_func_name+="product"
                    else:
                        acq_func_name+="sum"
                    
                    print(f"\n---------- Start {acq_func_name} training! ----------")
                    

                    avg_hist, avg_test=experiment_iter(args=args, device=device, datasets=datasets, acq_func=acq_func,state=state, beta=args.beta,
                        ratio=args.candidate_ratio,seeds=seeds,totest=True)

                    results[acq_func_name] = avg_hist
                    result_para[acq_func_name]=np.array([args.beta, avg_test])

            save_as_npy(data=results, folder=args.result_dir, name=acq_func_name)
            save_as_npy(data=result_para,folder=args.result_dir,name=acq_func_name+"para")

    else:

        print("This is two-stage query")

        if args.priority == 0:
            print("This is uncertainty first search")
        else:
            print("This is diversity first search")

        if args.candidate_ratio == 100:
            print("No ratio specified, conduct hyperparameter tuning")

            for state in state_loop:
                for i, acq_func in enumerate(acq_functions):
                    acq_func_name = str(acq_func).split(" ")[1] + "-MC=" + str(state)

                    if args.priority == 0:
                        acq_func_name+="uncertainty"
                    else:
                        acq_func_name+="diverisity"

                    if str(acq_func).split(" ")[1] == "uniform":
                        avg_hist, avg_test =experiment_iter(args=args, device=device, datasets=datasets, acq_func=acq_func,state=state, beta=args.beta,
                            ratio=args.candidate_ratio,seeds=seeds,totest=True)
                        results[acq_func_name] = avg_hist
                        result_para[acq_func_name]=np.array([avg_test])
                        break

                    ratio_test=[2, 3, 4, 5, 6]
                    ratio_test_value=[]

                    for ratio in ratio_test:
                        print('current ratio'+str(ratio))

                        avg_val=experiment_iter(args=args, device=device, datasets=datasets, acq_func=acq_func,state=state, beta=args.beta,
                            ratio=ratio,seeds=seeds, totest=False)

                        ratio_test_value.append(avg_val)

                    idx=np.argsort(-np.array(ratio_test_value))[0]
                    ratio_final=ratio_test[idx]
                    print('final ratio:'+str(ratio_final))

                    avg_hist, avg_test=experiment_iter(args=args, device=device, datasets=datasets, acq_func=acq_func,state=state, beta=args.beta,
                        ratio=ratio_final,seeds=seeds,totest=True)

                    results[acq_func_name] = avg_hist
                    result_para[acq_func_name]=np.array([ratio_final,avg_test])

            save_as_npy(data=results, folder=args.result_dir, name=acq_func_name)
            save_as_npy(data=result_para,folder=args.result_dir,name=acq_func_name+"para")

        else:
            print("ratio is specified, directly run the model")

            for state in state_loop:
                for i, acq_func in enumerate(acq_functions):

                    acq_func_name = str(acq_func).split(" ")[1] + "-MC=" + str(state) +"ratio_given"+str(args.candidate_ratio)

                    if args.priority == 0:
                        acq_func_name+="uncertainty"
                    else:
                        acq_func_name+="diverisity"
                    
                    print(f"\n---------- Start {acq_func_name} training! ----------")
                    

                    avg_hist, avg_test=experiment_iter(args=args, device=device, datasets=datasets, acq_func=acq_func,state=state, beta=args.beta,
                        ratio=args.candidate_ratio,seeds=seeds,totest=True)

                    results[acq_func_name] = avg_hist
                    result_para[acq_func_name]=np.array([args.candidate_ratio, avg_test])

            save_as_npy(data=results, folder=args.result_dir, name=acq_func_name)
            save_as_npy(data=result_para,folder=args.result_dir,name=acq_func_name+"para")


    print("--------------- Done Training! ---------------")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="EP",
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seed", type=int, default=369, metavar="S", help="random seed (default: 369)"
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=5,
        metavar="E",
        help="number of experiments (default: 3)",
    )
    parser.add_argument(
        "--dropout_iter",
        type=int,
        default=100,
        metavar="T",
        help="dropout iterations,T (default: 100)",
    )

    """
    The original codes set query times and dropout iteration times same, which should be controlled separately
    """
    parser.add_argument(
        "--query_times",
        type=int,
        default=20,
        metavar="QT",
        help="Times of query (default: 100)",
    )

    parser.add_argument(
        "--query",
        type=int,
        default=10,
        metavar="Q",
        help="number of query (default: 10)",
    )
    """
    Pool size for query
    """
    parser.add_argument(
        "--pool_size",
        type=int,
        default=2000,
        metavar="PS",
        help="pool size for query",
    )
    """
    Whether the weight of diversity should decay with time
    """
    parser.add_argument(
        "--time_decay",
        type=bool,
        default=False,
        metavar="TD",
        help="whether to decay weight with time",
    )

    """
    The combination of uncertainty and diverity metrics decides the query strategy
    """
    parser.add_argument(
        "--uncertainty",
        type=int,
        default=0,
        metavar="UN",
        help="uncertainty: 0 = entropy, 1 = bald, 2 = var_ratios, 10 = uniform, 100 = all ",
    )
    parser.add_argument(
        "--diversity",
        type=int,
        default=0,
        metavar="DI",
        help="diverisity: 0 = waal, 1 = density, 2 = minidis, 10 = uniform, 100 = all ",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=1000,
        metavar="V",
        help="validation set size (default: 100)",
    )
    parser.add_argument(
        "--determ",
        default=False,
        help="Compare with deterministic models (default: False)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="result_npy",
        metavar="SD",
        help="Save npy file in this folder (default: result_npy)",
    )
    """
    When using weighted product mode, the model will run with specified beta, otherwise run hyperparameter tuning
    """
    parser.add_argument(
        "--beta",
        type=float,
        default=100,
        help="specify beta")
    """
    Choose weighted product or two stage query method
    """
    parser.add_argument(
        "--runmode",
        type=int,
        default=0,
        help="whether to use weighted product or two stage query")

    """
    Choose look which metric first in two stage query 
    """

    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="priority of metrics: 0 = uncertainty first, 1 = diversity first")
    """
    Choose the ratio between candidate size and number of query in two stage query 
    """

    parser.add_argument(
        "--candidate_ratio",
        type=int,
        default=100,
        help="The ratio between candidates and number of query in two stage query")

    parser.add_argument(
        "--sum_product",
        type=int,
        default=1,
        help="Indicate whether using sum or product")


    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = dict()
    DataLoader = LoadData(args.val_size)
    (
        datasets["X_init"],
        datasets["y_init"],
        datasets["X_val"],
        datasets["y_val"],
        datasets["X_pool"],
        datasets["y_pool"],
        datasets["X_test"],
        datasets["y_test"],
    ) = DataLoader.load_all()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)


    results = train_active_learning(args, device, datasets)


if __name__ == "__main__":
    main()
