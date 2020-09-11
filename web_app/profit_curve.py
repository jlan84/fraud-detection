import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def profits(cf_lst, thresholds, cost_benefit_matrix):
    profit = []
    for i in range(len(thresholds)):
        profit.append(np.sum(cf_lst[i]*cost_benefit_matrix))
    return profit


if __name__ == "__main__":
    
    cost_benefit_matrix = np.array([[0,-1000],[-10000,-1000]])
    thresholds = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    x = np.array([[4293,12],[40,387]])
    total = np.sum(x)
    print(total)
    fraud = 387+40
    nf = 4293+12
    
    a = np.array([[4117, 188], [ 8, 419]])
    b = np.array([[4221, 84], [ 12, 415]])
    c = np.array([[4254, 51], [ 21, 406]])
    d = np.array([[4277, 28], [ 28, 399]])
    e = np.array([[4293, 12], [ 39, 388]])
    f = np.array([[4298, 7], [ 51, 376]])
    g = np.array([[4302, 3], [ 65, 362]])
    h = np.array([[4303, 2], [ 96, 331]])
    i = np.array([[4303, 2], [ 165, 262]])

    cf_mx_lst = [a,b,c,d,e,f,g,h,i]
    profit_lst = profits(cf_mx_lst, thresholds, cost_benefit_matrix)

    fig, ax = plt.subplots(figsize=(12,12))

    ax.plot(thresholds, profit_lst, scaley=True, color='green')
    ax.set_title('Profit vs Threshold Curve for Fraudulent Events')
    ax.set_ylabel('Profit mil$', fontsize=16)
    ax.set_xlabel('Threshold', fontsize=16)
    plt.ylim(-3000000,0)
    plt.tight_layout()
    plt.show()
