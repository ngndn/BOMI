import sys
import time
import math
import pickle
import datetime

import torch
import gpytorch
import pandas as pd
from scipy.stats import mode
from missingpy import KNNImputer
from sklearn.neighbors import KernelDensity

from ndfunction import *
from BOGPyTorch.GPTmodels import ExactGPModel
from MyBPMF.matrix_factorization import MyBPMF

parent_folder = \
    '/Users/nadan/google-drive/0-data/aalto/thesis/bayesian-optimization/software/BOMI/BOMissingInputs'
sys.path.append(parent_folder)


mySeed = 9
np.random.seed(mySeed)
torch.random.manual_seed(mySeed)


# Define BO class
class BOTorch:
    def __init__(self):
        return

    @staticmethod
    def optimize_hyperparameters(in_model, train_x, train_y, optimizer_algo, training_iter):
        # Set train data
        # train_x.cuda()
        # train_y.cuda()
        # in_model.cuda()
        in_model.set_train_data(train_x, train_y)
        # Find optimal model hyperparameters
        in_model.train()
        in_model.likelihood.train()
        # Use the adam optimizer
        if optimizer_algo == "Adam":
            optimizer = torch.optim.Adam([{'params': in_model.parameters()}, ],
                                         lr=0.1)  # Includes GaussianLikelihood parameters
        elif optimizer_algo == "SGD":
            optimizer = torch.optim.SGD([{'params': in_model.parameters()}, ],
                                        lr=0.1)  # Includes GaussianLikelihood parameters
        elif optimizer_algo == "LBFGS":
            optimizer = torch.optim.LBFGS([{'params': in_model.parameters()}, ], lr=0.1, history_size=50, max_iter=10,
                                          line_search_fn=True)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(in_model.likelihood, in_model)
        training_iter = training_iter

        # LBFGS
        # define closure
        if optimizer_algo == "LBFGS":
            def closure():
                optimizer.zero_grad()
                output = in_model(train_x)
                loss = -mll(output, train_y)
                return loss

            loss = closure()
            loss.backward()

            for i in range(training_iter):
                # perform step and update curvature
                loss = optimizer.step(closure)
                loss.backward()

        elif optimizer_algo == "SGD" or optimizer_algo == "Adam":
            # SGD & Adam
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = in_model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

    @staticmethod
    def posterior_predict(in_x, in_model):
        in_model.eval()
        in_model.likelihood.eval()
        # Calculate posterior predictions
        f_preds = in_model(in_x)
        # Take mean and variance
        tmp_mean = f_preds.mean
        tmp_var = f_preds.variance
        # Get from tensors
        mean = tmp_mean.detach().numpy()
        var = tmp_var.detach().numpy()
        return mean, var

    @staticmethod
    def posterior_predict_scalar(in_x, in_model):
        in_model.eval()
        in_model.likelihood.eval()
        mean = []
        var = []
        for point in in_x:
            # Push to tensor
            point = torch.tensor([point.item()])
            # Calculate posterior predictions
            f_preds = in_model(point)
            # Take mean and variance
            tmp_mean = f_preds.mean
            tmp_var = f_preds.variance
            # Get from tensors
            mean.append(tmp_mean.item())
            var.append(tmp_var.item())
        return mean, var

    @staticmethod
    def gpucb(in_x, in_model, in_Beta):
        in_model.eval()
        in_model.likelihood.eval()
        x = torch.tensor(in_x, dtype=torch.float32)
        # Calculate posterior predictions
        f_preds = in_model(x)
        # Take mean and variance
        tmp_mean = f_preds.mean
        tmp_var = f_preds.variance
        # Calculate acquisition
        tmp_acq = tmp_mean + torch.tensor(in_Beta) * tmp_var
        # Get from tensors
        acq = tmp_acq.clone().detach().cpu().numpy()
        tmp_mean = tmp_mean.clone().detach().cpu().numpy()
        tmp_var = tmp_var.clone().detach().cpu().numpy()
        return acq, tmp_mean, tmp_var

    @staticmethod
    def gpucb_scalar(in_x, in_model, in_beta):
        in_model.eval()
        in_model.likelihood.eval()
        for i in range(len(in_x)):
            in_x[i] = float(in_x[i])
        x = torch.tensor([in_x], dtype=torch.float32)
        # print("type x:", x)
        # Calculate posterior predictions
        f_preds = in_model(x)
        # Take mean and variance
        tmp_mean = f_preds.mean
        tmp_var = f_preds.variance
        # Calculate acquisition
        tmp_acq = tmp_mean + torch.tensor(in_beta) * tmp_var
        # Get from tensors
        acq = tmp_acq.detach().cpu().numpy()
        return acq


def run_experiment():
    # Input arguments
    input_method = sys.argv[1]  # method = "Drop", "Suggest", "BPMF", "BOMI", "Mean", "Mode", "KNN", "uGP"
    input_function = sys.argv[2]  # Black-box objective functions, see file 'ndfunction.py'
    input_num_gps = int(sys.argv[3])  # Default: 5 (GPs)
    input_in_alpha = float(sys.argv[4])  # Default: 1e2 - BPMF parameter
    input_missing_rate = float(sys.argv[5])  # 0.25
    input_missing_noise = float(sys.argv[6])  # 0.05

    # Select true objective function
    miss_rate = input_missing_rate
    miss_noise = input_missing_noise
    if input_function == "Eggholder2d":
        my_func = Eggholder(2, miss_rate, miss_noise, False)
    elif input_function == "Schubert4d":
        my_func = SchubertNd(4, miss_rate, miss_noise, False)
    elif input_function == "Alpine5d":
        my_func = AlpineNd(5, miss_rate, miss_noise, False)
    elif input_function == "Schwefel5d":
        my_func = SchwefelNd(5, miss_rate, miss_noise, False)

    # GP-UCB parameters
    scale_beta = 0.2
    delt = 0.1
    a = 1.0
    b = 1.0
    dim = my_func.input_dim
    r = 1.0

    BetaMu = 1.0
    BetaSigma = 1.0

    # Experiments settings
    Runs = 10
    if input_function == "Eggholder2d":
        num_iterations = 61
    elif input_function == "Schubert4d":
        num_iterations = 121
    elif input_function == "Alpine5d" or input_function == "Schwefel5d":
        num_iterations = 161

    log_iter = ""
    log_iter_point = ""
    log_run = ""

    # Select method
    method = input_method
    is_mul_method = True
    num_gps = input_num_gps
    gp_hypers_optim = "Adam"
    gp_optim_iter = 50
    total_time_start = time.time()
    for run in range(0, Runs):
        # Read data from file
        if input_function == "Eggholder2d":
            df_X = pd.read_csv(r'data/Eggholder2d/Eggholder2dX_' + str(run) + '.csv')
        elif input_function == "Schubert4d":
            df_X = pd.read_csv(r'data/Schubert4d/SchubertNd4dX_' + str(run) + '.csv')
        elif input_function == "Alpine5d":
            df_X = pd.read_csv(r'data/Alpine5d/AlpineNd5dX_' + str(run) + '.csv')
        elif input_function == "Schwefel5d":
            df_X = pd.read_csv(r'data/Schwefel5d/SchwefelNd5dX_' + str(run) + '.csv')

        in_X = df_X.values
        Xori = in_X.tolist().copy()

        if input_function == "Eggholder2d":
            dfY = pd.read_csv(r'data/Eggholder2d/EggholderY.csv')
        elif input_function == "Schubert4d":
            dfY = pd.read_csv(r'data/Schubert4d/SchubertNdY.csv')
        elif input_function == "Alpine5d":
            dfY = pd.read_csv(r'data/Alpine5d/AlpineNdY.csv')
        elif input_function == "Schwefel5d":
            dfY = pd.read_csv(r'data/Schwefel5d/SchwefelNdY.csv')

        inY = dfY.values
        inY = inY.squeeze()
        Yori = inY.tolist().copy()

        na = 0.0
        nb = 1.0
        minY = np.min(Yori)
        maxY = np.max(Yori)
        normalizedY = [[((nb - na) * (tmpY - minY) / (maxY - minY) + na)] for tmpY in Yori]

        normalizedX = [my_func.normalize(xi) for xi in Xori]

        R = np.append(normalizedX, normalizedY, axis=1)
        R[np.isnan(R)] = -1

        # Arrays of observations after dropping/removing
        dropNX = []
        dropNY = []
        dropYori = []
        dropXori = []
        for i in range(len(normalizedX)):
            if -1 not in R[i]:
                dropNX.append(normalizedX[i])
                dropNY.append(normalizedY[i])
                dropXori.append(Xori[i])
                dropYori.append(Yori[i])

        if method == "Drop" or method == "Suggest" or method == "uGP":
            in_X = dropXori
            in_Y = dropYori
            minY = np.min(in_Y)
            maxY = np.max(in_Y)
            if minY == maxY:
                minY -= 0.01
            n_in_Y = [((nb - na) * (tmpY - minY) / (maxY - minY) + na) for tmpY in in_Y]
            n_in_X = [my_func.normalize(xi) for xi in in_X]
        else:
            in_X = Xori
            in_Y = Yori
            n_in_X = normalizedX
            n_in_Y = normalizedY

        # Initialize likelihood and model
        train_x = torch.tensor(n_in_X, dtype=torch.float32)
        train_y = torch.tensor(n_in_Y, dtype=torch.float32)
        myGP = ExactGPModel(train_x, train_y)

        # Init BO objects:
        myBO = BOTorch()

        if method == "BPMF" or method == "BOMI":
            R_in = np.append(n_in_X, n_in_Y, axis=1)
            R_in[np.isnan(R_in)] = -1
            D = 15
            N, M = R.shape
            T = 40

            beta0 = None
            my_bpmf_object = MyBPMF()

        in_Guessed_X = dropXori
        in_Guessed_Y = dropYori
        in_Guessed_nY = [((nb - na) * (tmpY - np.min(in_Guessed_Y)) / (np.max(in_Guessed_Y) - np.min(in_Guessed_Y)) + na)
                         for tmpY in in_Guessed_Y]
        in_Guessed_nX = [my_func.normalize(xi) for xi in in_Guessed_X]

        # BO loops
        my_func.numCalls = 0
        BOstart_time = time.time()
        for ite in range(num_iterations):
            precalBetaT = 2.0 * np.log((ite + 1) * (ite + 1) * math.pi ** 2 / (3 * delt)) + 2 * dim * np.log(
                (ite + 1) * (ite + 1) * dim * b * r * np.sqrt(np.log(4 * dim * a / delt)))
            BetaT = np.sqrt(precalBetaT) * scale_beta

            # Train the GP model
            # BPMF require filling missing values first before feeding into the GP
            if method == "BPMF":
                R_in = np.append(n_in_X, n_in_Y, axis=1)
                R_in[np.isnan(R_in)] = -1
                new_X = []
                new_Y = []

                (N, M) = R_in.shape
                U_in = np.zeros((D, N))
                V_in = np.zeros((D, M))

                in_alpha = input_in_alpha * r
                R_pred, train_err_list, Rs = my_bpmf_object.proposed_bpmf(R_in, R_in, U_in, V_in, T, D,
                                                                          initial_cutoff=0, lowest_rating=0.0,
                                                                          highest_rating=1.0, in_alpha=in_alpha,
                                                                          numSamples=num_gps,
                                                                          Beta_0=beta0, output_file=None,
                                                                          missing_mask=-1,
                                                                          save_file=False)
                # Use the new predicted matrix as input training data for the GP
                res = R_pred.copy()
                new_X = np.delete(res, my_func.input_dim, axis=1)
                new_Y = (np.delete(res, 0, axis=1)).tolist()
                for _ in range(my_func.input_dim - 2):
                    new_Y = (np.delete(new_Y, 0, axis=1)).tolist()
                new_Y = ((np.delete(new_Y, 0, axis=1)).reshape(1, -1)).tolist()

                train_x = torch.tensor(new_X, dtype=torch.float32)
                train_y = torch.tensor(new_Y, dtype=torch.float32)

                myGP = ExactGPModel(train_x, train_y)
                myBO.optimize_hyperparameters(myGP, train_x, train_y, gp_hypers_optim, gp_optim_iter)
            elif method == "BOMI":
                BPMFstart_time = time.time()

                R_in = np.append(n_in_X, n_in_Y, axis=1)
                R_in[np.isnan(R_in)] = -1

                new_X = []
                new_Y = []

                (N, M) = R_in.shape
                U_in = np.zeros((D, N))
                V_in = np.zeros((D, M))

                in_alpha = input_in_alpha*r
                R_pred, train_err_list, Rs = my_bpmf_object.proposed_bpmf(R_in, R_in, U_in, V_in, T, D,
                                                                          initial_cutoff=0, lowest_rating=0.0,
                                                                          highest_rating=1.0, in_alpha=in_alpha,
                                                                          numSamples=num_gps,
                                                                          Beta_0=beta0, output_file=None,
                                                                          missing_mask=-1,
                                                                          save_file=False)

                BPMFstop_time = time.time()

                PRstart_time = time.time()
                idxs = np.where(R_in == -1)
                for ii in range(num_gps):
                    tmpR = R_in.copy()
                    for iii in range(len(idxs[0])):
                        tmpR[idxs[0][iii], idxs[1][iii]] = Rs[ii][idxs[0][iii]][idxs[1][iii]]
                    Rs[ii] = tmpR

                PRstop_time = time.time()

                # Use the new predicted matrix as input training data for the GP
                BuildGPstart_time = time.time()
                GPs = []
                for ii in range(num_gps):
                    res = Rs[ii].copy()
                    new_X = np.delete(res, my_func.input_dim, axis=1)
                    new_Y = (np.delete(res, 0, axis=1)).tolist()
                    for _ in range(my_func.input_dim - 2):
                        new_Y = (np.delete(new_Y, 0, axis=1)).tolist()
                    new_Y = ((np.delete(new_Y, 0, axis=1)).reshape(1, -1)).tolist()

                    train_x = torch.tensor(new_X, dtype=torch.float32)
                    train_y = torch.tensor(new_Y, dtype=torch.float32)
                    tmpGP = ExactGPModel(train_x, train_y)

                    myBO.optimize_hyperparameters(tmpGP, train_x, train_y, gp_hypers_optim, gp_optim_iter)
                    GPs.append(tmpGP)

                BuildGPstop_time = time.time()
            elif method == "Mean":

                R_in = np.append(n_in_X, n_in_Y, axis=1)
                R_in[np.isnan(R_in)] = -1
                idxs = np.where(R_in == -1)

                tmpR = R_in.copy()
                meansR = np.mean(tmpR, axis=0)
                for iii in range(len(idxs[0])):
                    tmpR[idxs[0][iii], idxs[1][iii]] = meansR[idxs[1][iii]]

                # Use the new predicted matrix as input training data for the GP
                BuildGPstart_time = time.time()
                GPs = []

                res = tmpR

                new_X = np.delete(res, my_func.input_dim, axis=1)
                new_Y = (np.delete(res, 0, axis=1)).tolist()

                n_in_X = (new_X.tolist()).copy()
                n_in_Y = new_Y.copy()

                for _ in range(my_func.input_dim - 2):
                    new_Y = (np.delete(new_Y, 0, axis=1)).tolist()
                new_Y = ((np.delete(new_Y, 0, axis=1)).reshape(1, -1)).tolist()

                train_x = torch.tensor(new_X, dtype=torch.float32)
                train_y = torch.tensor(new_Y, dtype=torch.float32)
                myGP = ExactGPModel(train_x, train_y)
                myBO.optimize_hyperparameters(myGP, train_x, train_y, gp_hypers_optim, gp_optim_iter)

                BuildGPstop_time = time.time()
                print('Built GP Completed')
                print("Build GPs time:", str(BuildGPstop_time - BuildGPstart_time), " seconds")
            elif method == "Mode":

                R_in = np.append(n_in_X, n_in_Y, axis=1)
                RforIdx = R_in.copy()
                RforIdx[np.isnan(RforIdx)] = -1

                idxs = np.where(RforIdx == -1)

                tmpR = R_in.copy()
                modesR = mode(tmpR, axis=0, nan_policy='omit')[0][0]
                for iii in range(len(idxs[0])):
                    tmpR[idxs[0][iii], idxs[1][iii]] = modesR[idxs[1][iii]]

                # Use the new predicted matrix as input training data for the GP
                BuildGPstart_time = time.time()
                GPs = []

                res = tmpR

                new_X = np.delete(res, my_func.input_dim, axis=1)
                new_Y = (np.delete(res, 0, axis=1)).tolist()
                for _ in range(my_func.input_dim - 2):
                    new_Y = (np.delete(new_Y, 0, axis=1)).tolist()
                new_Y = ((np.delete(new_Y, 0, axis=1)).reshape(1, -1)).tolist()

                n_in_X = (new_X.tolist()).copy()
                n_in_Y = new_Y.copy()

                train_x = torch.tensor(new_X, dtype=torch.float32)
                train_y = torch.tensor(new_Y, dtype=torch.float32)
                myGP = ExactGPModel(train_x, train_y)
                myBO.optimize_hyperparameters(myGP, train_x, train_y, gp_hypers_optim, gp_optim_iter)

                BuildGPstop_time = time.time()
                print('Built GP Completed')
                print("Build GPs time:", str(BuildGPstop_time - BuildGPstart_time), " seconds")
            elif method == "KNN":
                R_in = np.append(n_in_X, n_in_Y, axis=1)
                RforIdx = R_in.copy()
                RforIdx[np.isnan(RforIdx)] = -1

                idxs = np.where(RforIdx == -1)

                tmpR = RforIdx.copy()
                imputer = KNNImputer(missing_values=-1, n_neighbors=3, weights="uniform")
                knnR = imputer.fit_transform(tmpR)

                for iii in range(len(idxs[0])):
                    tmpR[idxs[0][iii], idxs[1][iii]] = knnR[idxs[0][iii], idxs[1][iii]]

                # Use the new predicted matrix as input training data for the GP
                BuildGPstart_time = time.time()
                GPs = []

                res = tmpR

                new_X = np.delete(res, my_func.input_dim, axis=1)
                new_Y = (np.delete(res, 0, axis=1)).tolist()
                for _ in range(my_func.input_dim - 2):
                    new_Y = (np.delete(new_Y, 0, axis=1)).tolist()
                new_Y = ((np.delete(new_Y, 0, axis=1)).reshape(1, -1)).tolist()

                n_in_X = (new_X.tolist()).copy()
                n_in_Y = new_Y.copy()

                train_x = torch.tensor(new_X, dtype=torch.float32)
                train_y = torch.tensor(new_Y, dtype=torch.float32)
                myGP = ExactGPModel(train_x, train_y)
                myBO.optimize_hyperparameters(myGP, train_x, train_y, gp_hypers_optim, gp_optim_iter)

                BuildGPstop_time = time.time()
                print('Built GP Completed')
                print("Build GPs time:", str(BuildGPstop_time - BuildGPstart_time), " seconds")
            elif method == "uGP":
                if input_function == "HeatTreatment" or input_function == "RobotSim":
                    for idxPoint in range(len(n_in_X)):
                        for idx2Point in range(idxPoint+1, len(n_in_X)):
                            checkdis = np.linalg.norm(np.array(n_in_X[idxPoint]) - np.array(n_in_X[idx2Point]),2)
                            if checkdis < 1.1:
                                if n_in_Y[idxPoint] >= n_in_Y[idx2Point]:
                                    n_in_X.remove(n_in_X[idx2Point])
                                    in_X.remove(in_X[idx2Point])
                                    n_in_Y.remove(n_in_Y[idx2Point])
                                    in_Y.remove(in_Y[idx2Point])
                                else:
                                    n_in_X.remove(n_in_X[idxPoint])
                                    n_in_Y.remove(n_in_Y[idxPoint])
                                    in_X.remove(in_X[idxPoint])
                                    in_Y.remove(in_Y[idxPoint])
                                break

                # instantiate and fit the KDE model
                kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
                kde.fit(n_in_X)

                # score_samples returns the log of the probability density
                logprobs = kde.score_samples(n_in_X)

                # Use the new predicted matrix as input training data for the GP
                BuildGPstart_time = time.time()
                GPs = []

                new_X = (logprobs.tolist()).copy()
                new_Y = n_in_Y.copy()

                train_x = torch.tensor(new_X, dtype=torch.float32)
                train_y = torch.tensor(new_Y, dtype=torch.float32)
                myGP = ExactGPModel(train_x, train_y)
                myBO.optimize_hyperparameters(myGP, train_x, train_y, gp_hypers_optim, gp_optim_iter)
            else:
                train_x = torch.tensor(n_in_X, dtype=torch.float32)
                train_y = torch.tensor(n_in_Y, dtype=torch.float32)
                myBO.optimize_hyperparameters(myGP, train_x, train_y, gp_hypers_optim, gp_optim_iter)

            log_iter += str(np.max(np.union1d(in_Y, Yori))) + '\n'
            # Strategy for choosing the next points
            if method == "Suggest":
                testX = [my_func.rand_uniform_in_nbounds() for i in range(10000)]
                testX = torch.tensor(testX)
                # Set into posterior mode
                myGP.eval()
                myGP.likelihood.eval()

                # TS
                # preds = myGP.likelihood(myGP(testX))
                # newSample = preds.sample()
                # newSample = newSample.cpu()
                # nextX = testX[np.argmax(newSample)]
                # nextX = nextX.detach().cpu().numpy()

                # GPUCB
                acq, _, _ = myBO.gpucb(testX, myGP, BetaT)
                # acqs.append(acq.tolist())
                next_X = testX[np.argmax(acq)]
                next_X = next_X.detach().cpu().numpy()
            elif method == "BOMI":
                opt_sug_starttime = time.time()
                # Sample the next point from each GP
                testX = [my_func.rand_uniform_in_nbounds() for i in range(10000)]

                candidatesX = testX.copy()
                testX = torch.tensor(testX)
                nextXs = []
                next_Xs_acq = []

                acqs = []
                for ii in range(num_gps):
                    GPs[ii].eval()
                    GPs[ii].likelihood.eval()
                    acq, mean_i, var_i = myBO.gpucb(testX, GPs[ii], BetaT)
                    acqs.append(acq)

                sAcq = np.mean(acqs, axis=0) + BetaT*np.std(acqs, axis=0)

                tmpX = testX[np.argmax(sAcq)]
                tmpX = tmpX.detach().cpu().numpy()
                nextXs.append(tmpX.tolist())
                next_Xs_acq.append(sAcq[np.argmax(sAcq)])

                next_X = nextXs[np.argmax(next_Xs_acq)]

                opt_sug_stoptime = time.time()
                print("method:", method, "next X:", next_X, " idx:", np.argmax(next_Xs_acq))
            elif method == "Mean":
                # Random sampling
                testX = [my_func.rand_uniform_in_nbounds() for i in range(10000)]
                testX = torch.tensor(testX)

                acq, _, _ = myBO.gpucb(testX, myGP, BetaT)

                next_X = testX[np.argmax(acq)]
                next_X = next_X.detach().cpu().numpy()
            elif method == "uGP":
                testX = [my_func.rand_uniform_in_nbounds() for i in range(10000)]
                logprobsX = kde.score_samples(testX)
                probX = torch.tensor(logprobsX)

                acq, _, _ = myBO.gpucb(probX, myGP, BetaT)

                next_X = testX[np.argmax(acq)]
                print("method:", method, "nextX:", next_X)
            else:
                # Random sampling
                testX = [my_func.rand_uniform_in_nbounds() for i in range(10000)]
                testX = torch.tensor(testX)

                acq, _, _ = myBO.gpucb(testX, myGP, BetaT)

                next_X = testX[np.argmax(acq)]
                next_X = next_X.detach().cpu().numpy()

            # Query true objective function
            nextY, out_X = my_func.func_with_missing(my_func.denormalize(next_X))
            print("Out X:", out_X)

            # Argument data and update the statistical model
            if method == "Drop":
                if np.isnan(out_X).any():
                    # We skip the observation if there is a missing value
                    print("Drop!!")
                    continue
                else:
                    # Add to D (DropBO method)
                    in_X.append(my_func.denormalize(next_X))
                    n_in_X.append(next_X)
                    in_Y.append(nextY)
            elif method == "Suggest" or method == "uGP":
                in_X.append(my_func.denormalize(next_X))
                n_in_X.append(next_X)
                in_Y.append(nextY)
            else:
                # Add to D
                nout_X = my_func.normalize(out_X)
                in_X.append(my_func.denormalize(nout_X))
                n_in_X.append(nout_X)
                in_Y.append(nextY)

            print("Next X:", my_func.denormalize(next_X), " next Y:", nextY)
            minY = np.min(in_Y)
            maxY = np.max(in_Y)
            if method == "BPMF":
                n_in_Y = [[((nb - na) * (tmpY - minY) / (maxY - minY) + na)] for tmpY in in_Y]
            elif method == "BOMI" or method == "Mean" or method == "Mode" or method == "KNN":
                n_in_Y = [[((nb - na) * (tmpY - minY) / (maxY - minY) + na)] for tmpY in in_Y]
                in_Guessed_nY = [((nb - na) * (tmpY - np.min(in_Guessed_Y)) /
                                  (np.max(in_Guessed_Y) - np.min(in_Guessed_Y)) + na)
                                 for tmpY in in_Guessed_Y]
            else:
                if method == "Suggest" or method == "Drop" or method == "uGP":
                    if minY == maxY:
                        minY -= 0.01

                n_in_Y = [((nb - na) * (tmpY - minY) / (maxY - minY) + na) for tmpY in in_Y]

                train_x = torch.tensor(n_in_X, dtype=torch.float32)
                train_y = torch.tensor(n_in_Y, dtype=torch.float32)

                myGP = ExactGPModel(train_x, train_y)
            print("Iter ", ite, " Optimum Y: \033[1m", np.max(np.union1d(in_Y, Yori)), "\033[0;0m at: ",
                  my_func.denormalize(n_in_X[np.argmax(in_Y)]))
            print()

        BOstop_time = time.time()
        ymax = np.max(np.union1d(in_Y, Yori))
        print("Run:", run, " method: ", method, " y: ", str(ymax), " numCalls: ", my_func.numCalls, " ite:", ite,
              " time: --- %s seconds ---" % (BOstop_time - BOstart_time))
        print()

        # END BO loops

    print("Solution: x=", in_X[np.argmax(in_Y)], " f(x)=", np.max(in_Y))


if __name__ == '__main__':
    run_experiment()
