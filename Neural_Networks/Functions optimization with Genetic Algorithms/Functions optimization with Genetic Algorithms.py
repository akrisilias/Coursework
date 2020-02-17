#!/usr/bin/env python
# coding: utf-8

# ## Στοιχεία Ομάδας
# ### Ομάδα Α9
# <br>
# Μαύρος Γεώργιος<br>03112618<br><br>
# Κρίσιλιας Ανδρέας<br>03114778 

# ## Βασικά install & import

# In[1]:


# install

get_ipython().system(u' pip install deap')


# In[1]:


# import

from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# ## Συναρτήσεις προς βελτιστοποίηση

# Ορίζουμε αρχικά τις συναρτήσεις που έχουμε προς βελτιστοποίηση.<br><br>
# Η μη-κλιμακούμενη είναι η <b>Mishra 9 Function</b>
# <p>$$f(X)=\left[f_1 f^2_2 f_3 + f_1 f_2 f^2_3 + f^2_2 +\left(x_1+x_2-x_3\right)^2\right]^2$$</p>
# <p>where:</p>
# <p>\(\bullet\) \(f_1=2x^3_1+5x_1x_2+4x_3-2x^2_1x_3-18\)</p>
# <p>\(\bullet\) \(f_2=x_1+x^3_2+x_1x^2_2+x_1x^2_3-22\)</p>
# <p>\(\bullet\) \(f_3=8x^2_1+2x_2x_3+2x^2_2+3x^3_2-52\)</p>
# <p>\(\bullet\) \(f_{min}(X^*)=0\)</p>
# <p>\(\bullet\) \(x^*_i=(1,2,3)\)</p>
# <br>
# <br>Και η κλιμακούμενη είναι η <b>Schwefel 2.22 Function</b><br>
# <p>$$f(\mathbf{x})=f(x_1, ..., x_n)=\sum_{i=1}^{n}|x_i|+\prod_{i=1}^{n}|x_i|$$</p>
# <p>\(\bullet\) $f(\textbf{x}^{\ast})=0$ at $\textbf{x}^{\ast} = (0, …, 0)$.</p>
# <p>\(\bullet\) $x_i \in [-100, 100]$ for $i=1, …, n$.</p>

# In[39]:


# Mishra 9 Function
def mishra9(indiv):
    a = 2*indiv[0]**3 + 5*indiv[0]*indiv[1] + 4*indiv[2] - 2*indiv[0]**2*indiv[2] - 18
    b = indiv[0] + indiv[1]**3 + indiv[0]*indiv[2]**2 - 22
    c = 8*indiv[0]**2 + 2*indiv[1]*indiv[2] + 2*indiv[1]**2 + 3*indiv[1]**3 -52
    total = (a*b**2*c + a*b*c**2 + b**2 + (indiv[0] + indiv[1] - indiv[2])**2)**2
    return (total,)


# In[2]:


# Schwefel 2.22 Function
def schwefel2_22(indiv):
    sum_part = 0
    pro_part = 1
    for ind in indiv:
        sum_part += abs(ind)
        pro_part *= abs(ind)
    total = sum_part + pro_part
    return (total,)


# ## Αξιολόγηση Γενετικών Αλγορίθμων

# <p>Αρχικά θα ορίσουμε μία συνάρτηση για την εκτέλεση των γενετικών αλγορίθμων.</p>
# <p>Ως ορίσματα θα δέχεται τον αριθμό γενεών, το πλήθος γονιδίων, τη συνάρτηση προς ελαχιστοποίηση, τους τελεστές διασταύρωσης και μετάλλαξης μαζί με τα ορίσματά τους, τα ορίσματα για τον selTournament, τον αλγόριθμο εξέλιξης με τα ορίσματά του και τις πιθανότητες διασταύρωσης και μετάλλαξης.</p>
# <p>Ως έξοδο θα επιστρέφει ένα LogBook με τα στατιστικά για κάθε γενεά, καθώς επίσης και το χρόνο εκτέλεσης του αλγορίθμου.</p>

# In[3]:


# function to execute Evolutionary Algorithms (ea)
def ea_func(gens,numVariables, evalFunc, cxAlgo, cxArgs, mutAlgo, mutArgs, selArgs, eaAlgo, eaArgs, popul, cxpb, mutpb):
    
    # we want to minimize our function
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create( "IndividualContainer", list , fitness= creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # gonna search in range [-100,100)
    toolbox.register( "InitialValue", np.random.uniform, -10., 10.)
    toolbox.register( "indiv", tools.initRepeat, creator.IndividualContainer, toolbox.InitialValue, numVariables)
    toolbox.register( "population", tools.initRepeat, list , toolbox.indiv)
    
    toolbox.register( "evaluate", evalFunc)
    # only our scalable function has domain limitations
    if (evalFunc.__name__ == 'schwefel2_22'):
        
        MIN_BOUND = np.array([-100.]*numVariables)
        MAX_BOUND = np.array([100.]*numVariables)

        def feasible( indiv ):
            if any( indiv < MIN_BOUND) or any( indiv > MAX_BOUND):
                return False
            return True

        def distance( indiv ) :
            dist = 0.0
            for i in range (len( indiv )) :
                penalty = 0
                if ( indiv [i] < MIN_BOUND[i]) : penalty = -100. - indiv [i]
                if ( indiv [i] > MAX_BOUND[i]) : penalty = indiv [i] - 100.
                dist = dist + penalty
            return dist
        
        # we set the function max as constant penalty (here named foul)
        foul = 100*numVariables + 100**numVariables
        toolbox.decorate( "evaluate", tools.DeltaPenality (feasible, foul, distance))
    
    toolbox.register( "mate", cxAlgo, **cxArgs)
    toolbox.register( "mutate", mutAlgo, **mutArgs)
    toolbox.register( "select", tools.selTournament, **selArgs)
    
    pop = toolbox.population(n=popul)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    start = time.time()
    pop, logbook = eaAlgo(pop, toolbox, ngen=gens, cxpb=cxpb, mutpb=mutpb, stats=stats, halloffame=hof, verbose=False, **eaArgs)
    end = time.time()
    
    ex_time = end - start
    
    return logbook, ex_time, hof


# <p>Στη συνέχεια θα ορίσουμε μία συνάρτηση που θα υπολογίζει τα απόλυτα και τα σχετικά κριτήρια για ένα συγκεκριμένο αλγόριθμο καλώντας την παραπάνω συνάρτηση έναν αριθμό γύρων.</p>
# <p>Τα ορίσματα αυτής της συνάρτησης θα έιναι επομένως τα ίδια με της προηγούμενης, με έξτρα τον αριθμό DELTA και το πλήθος των γύρων. Ο αριθμός GOAL δεν περνάει ως όρισμα, καθώς στην προκειμένη περίπτωση είναι 0 για αφμότερες τις συναρτήσεις προς βελτιστοποίηση.</p>
# <p>Ως έξοδο θα επιστρέφει τα υπολογισμένα κριτήρια για το συγκεκριμένο αλγόριθμο, δηλαδή τα:<br>
# successes, s.avg.min, s.avg.evals, s.avg.gens, avg.evals, avg.min, avg.time</p>

# In[4]:


# function that calculates absolute and relative
# criteria for a single ea algorithm
def singleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgo, cxArgs, mutAlgo, mutArgs, selArgs, eaAlgo, eaArgs, popul, cxpb, mutpb):
    # goal is 0 for both functions we've got
    # ain't pass it as argument
    
    logbooks = []
    times = []
    
    for i in range (0, rounds):
        l, t, _ = ea_func(gens,numVariables, evalFunc, cxAlgo, cxArgs, mutAlgo, mutArgs, selArgs, eaAlgo, eaArgs, popul, cxpb, mutpb)
        logbooks.append(l)
        times.append(t)
    
    mins = []
    evals = 0
    
    goal_delta = False
    s_gens = []
    s_mins = []
    s_evals = []
    
    for i in range (0, rounds):
        logbook = logbooks[i]
        log_mins = logbook.select('min')
        # append min of each round to mins
        mins.append(min(log_mins))
        # add all evals of round to evals
        evals += sum(logbook.select("nevals"))
        # if successfully under goal + delta (= 0 + delta = delta)..
        for j in range (0, gens+1):
            if (logbook[j]['min'] < delta):
                s_gens.append(j)
                s_mins.append(logbook[j]['min'])
                s_evals.append(sum(logbook.select('nevals')[0:j+1]))
                goal_delta = True
                break
        
    avg_min = sum(mins) / rounds
    avg_evals = evals / rounds
    avg_time = sum(times) / rounds
    successes = sum(m<delta for m in mins)
    if (goal_delta):
        s_avg_gens = sum(s_gens) / len(s_gens)
        s_avg_min = sum(s_mins) / len(s_mins)
        s_avg_evals = sum(s_evals) / len(s_evals) 
    else:
        s_avg_gens = 'None'
        s_avg_min = 'None'
        s_avg_evals = 'None'
        
    return [successes, s_avg_min, s_avg_evals, s_avg_gens, avg_evals, avg_min, avg_time]
    


# <p>Ορίζουμε τώρα μία συνάρτηση που υπολογίζει συγκεντρωτικά τα σχετικά και απόλυτα κριτήρια για τους διάφορους συνδυασμούς γενετικών τελεστών - στρατηγικών εξέλιξης καλώντας την αποπάνω για κάθε συνδυασμό.</p>
# <p>Τα ορίσματά της είναι τα ίδια με της προηγούμενης συνάρτησης, με τη διαφορά ότι πλέον δέχεται λίστα από γενετικούς τελεστές και πιθανότητες, λίστα από αλγορίθμους εξέλιξης και λίστες από ορίσματα αυτών.</p>
# <p>Ως έξοδος επιστρέφεται ένα DataFrame με τα υπολογισμένα κριτήρια για κάθε συνδυασμό τελεστών - στρατηγικών.</p>

# In[5]:


def multipleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM):
    headers = ['operators', 'strategy', 'successes', 's.avg.min', 's.avg.evals', 's.avg.gens', 'avg.evals', 'avg.min', 'avg.time']
    table = []
    for c1 in range (0, len(cxAlgoM)):
        for c2 in range (2*c1, min(2*c1+2,len(cxArgsM))):
            for m1 in range (0, len(mutAlgoM)):
                for m2 in range (2*m1, min(2*m1+2,len(mutArgsM))):
                    for s in range (0, len(selArgsM)):
                        for e in range (0, len(eaAlgoM)):
                            for cxpb in cxpbM:
                                for mutpb in mutpbM:
                                    # Operators
                                    oper_str = ""
                                    cxAlgo = cxAlgoM[c1]
                                    oper_str += cxAlgo.__name__
                                    cxArgs = cxArgsM[c2]
                                    if (len(cxArgsM) == 1):
                                        oper_str += ""
                                    elif (c2%2 == 0):
                                        oper_str += "High,"
                                    else:
                                        oper_str += "Low,"
                                    # same for mut
                                    mutAlgo = mutAlgoM[m1]
                                    oper_str += mutAlgo.__name__
                                    mutArgs = mutArgsM[m2]
                                    if (len(mutArgsM) == 1):
                                        oper_str += ""
                                    elif (m2%2 == 0):
                                        oper_str += "High,"
                                    else:
                                        oper_str += "Low,"
                                    # now for sel
                                    oper_str += "Selection"
                                    selArgs = selArgsM[s]
                                    if (len(selArgsM) == 1):
                                        oper_str += ""
                                    elif (s%2 == 0):
                                        oper_str += "High"
                                    else:
                                        oper_str += "Low"
                                    # Strategy
                                    stra_str = ""
                                    eaAlgo = eaAlgoM[e]
                                    eaArgs = eaArgsM[e]
                                    if (eaAlgo.__name__ == "eaSimple"):
                                        stra_str += eaAlgo.__name__ + " " + str(popul) + " " + str(cxpb) + " " + str(mutpb)
                                    else:
                                        stra_str += eaAlgo.__name__ + " " + str(eaArgs['mu']) + " " + str(eaArgs['lambda_']) + " " + str(cxpb) + " " + str(mutpb)
                                    # now let's create a row for the table
                                    row = [oper_str, stra_str]
                                    stats = singleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgo, cxArgs, mutAlgo, mutArgs, selArgs, eaAlgo, eaArgs, popul, cxpb, mutpb)
                                    row.extend(stats)
                                    # and append the row to the table
                                    table.append(row)
    # table should be ready by now
    tableDF = pd.DataFrame(table, columns=headers)
    # return table
    return tableDF


# ## Μέρος 1. Βελτιστοποίηση μη κλιμακούμενης συνάρτησης
# <br><center><b>Mishra 9 Function</b><center><br>
# <p>$$f(X)=\left[f_1 f^2_2 f_3 + f_1 f_2 f^2_3 + f^2_2 +\left(x_1+x_2-x_3\right)^2\right]^2$$</p>
# <p>where:</p>
# <p>\(\bullet\) \(f_1=2x^3_1+5x_1x_2+4x_3-2x^2_1x_3-18\)</p>
# <p>\(\bullet\) \(f_2=x_1+x^3_2+x_1x^2_2+x_1x^2_3-22\)</p>
# <p>\(\bullet\) \(f_3=8x^2_1+2x_2x_3+2x^2_2+3x^3_2-52\)</p>
# <p>\(\bullet\) \(f_{min}(X^*)=0\)</p>
# <p>\(\bullet\) \(x^*_i=(1,2,3)\)</p>

# ### Εύρεση βέλτιστου συνδυασμού τελεστών - στρατηγικής

# #### Γενετικοί τελεστές
# <p>Οι τελεστές <b>διαστραύρωσης</b> που επιλέξαμε είναι οι cxBlend και cxSimulatedBinary.</p>
# <p>\(\bullet\) Για τον πρώτο επιλέγουμε για υπερπαράμετρο high: {'alpha':0.7}, ενώ για low: {'alpha':0.3}</p>
# <p>\(\bullet\) Για το δεύτερο επιλέγουμε για υπερπαράμετρο high: {'eta':1}, ενώ για low: {'eta':5}</p><br>
# <p>Οι τελεστές <b>μετάλλαξης</b> που επιλέξαμε είναι οι mutGaussian και mutShuffleIndexes.</p>
# <p>\(\bullet\) Για τον πρώτο επιλέγουμε για συνδυασμό υπερπαραμέτρων high: {'mu':0, 'sigma':5, 'indpb':1./numVariables}, ενώ για low: {'mu':0, 'sigma':1, 'indpb':1./numVariables}</p>
# <p>\(\bullet\) Για το δεύτερο επιλέγουμε για υπερπαράμετρο high: {'indpb':1.3/numVariables}, ενώ για low: {'indpb':0.7/numVariables}</p>
# <p>* όπου numVariables το πλήθος των γονιδίων ανά χρωμόσωμα, στην προκειμένη numVariables = 3</p><br>
# <p>Για τον τελεστή <b>επιλογής</b> selTournament επιλέγουμε για υπερπαράμετρο high: {'tournsize':2}, ενώ για low: {'tournsize':10}</p>

# In[56]:


cxAlgoM = [tools.cxBlend, tools.cxSimulatedBinary]
cxArgsM = [{'alpha':0.7}, {'alpha':0.3}, {'eta':1}, {'eta':5}]
mutAlgoM = [tools.mutGaussian, tools.mutShuffleIndexes]
mutArgsM = [{'mu':0, 'sigma':5, 'indpb':1./numVariables}, {'mu':0, 'sigma':1, 'indpb':1./numVariables}, {'indpb':1.3/numVariables}, {'indpb':0.7/numVariables}]
selArgsM = [{'tournsize':2}, {'tournsize':10}]


# #### Στρατηγική εξέλιξης
# <p>Οι τρεις εξελικτικοί αλγόριθμοι που θα χρησιμοποιηθούν είναι οι eaSimple, eaMuPlusLambda, eaMuCommaLambda</p>
# <p>Για τιμή του <b>μ</b> θα θέσουμε το ένα τέταρτο του πληθυσμού, ενώ για τιμή του <b>λ</b> τα τρία τέταρτα του πληθυσμού.</p>

# In[57]:


eaAlgoM = [algorithms.eaSimple, algorithms.eaMuPlusLambda, algorithms.eaMuCommaLambda]
popul = 280 # population
eaArgsM = [{}, {'mu':popul*1/4, 'lambda_':popul*3/4}, {'mu':popul*1/4, 'lambda_':popul*3/4}]


# #### Μεθοδολογία εύρεσης βέλτιστου συνδυασμού
# <p>Διαλέγουμε ένα σχετικά μικρό αριθμό γύρων (10) και μέγιστων γενεών (150).</p>
# <p>Θέτουμε σταθερή τιμή στην παράμετρο πληθυσμού (280) και στις πιθανότητες διασταύρωσης (0.8) και μετάλλαξης (0.2).</p>
# <p>Για το DELTA επιλέγουμε τιμή κοντά στο ολικό ελάχιστο του καλύτερου συνδυασμού (2.1e-29).</p>

# In[58]:


# find limit so that only 10 combinations have avg.min below DELTA
sum(stats_table['avg.min'] < 2.1e-29)
# execution has been done in advance from cell down below


# In[101]:


rounds = 10
gens = 150
cxpbM = [0.8]
mutpbM = [0.2]
delta = 2.1e-29


# #### Τελική αναζήτηση και σχολιασμός
# <p>Θέτουμε ως συνάρτηση προς βελτιστοποίηση τη Mishra 9 Function και τρέχουμε την τελική αναζήτηση.</p>
# <p>Οι τιμές που επιλέξαμε για πληθυσμό είναι τέτοιες ώστε να βγαίνει το avg.time γύρω στο 2 τα τους πιο αργούς συνδυασμούς. Ξεκινήσαμε από 150 και είδαμε ότι τρέχει πολύ γρήγορα ο κάθε γύρος. Αυξήσαμε σε 300 και σταδιακά αυξομειώσαμε μέχρι να καταλήξουμε στην τελική τιμή μας (280).</p>
# <p>Για τιμή του DELTA, όπως φάνηκε και παραπάνω, βρήκαμε μία τιμή (2.1e-29) η οποία να είναι μεγαλύτερη από ακριβώς 10 συνδυασμών avg.min (για δεδομένη πλέον τιμή πληθυσμού), ώστε μόνο οι καλύτεροι συνδυασμοί να είναι κοντά στη μονάδα σε ποσοστό επιτυχιών.</p>
# <p>Οι τιμές για τις πιθανότητες διασταύρωσης και μετάλλαξης είναι αρχικά κοντά σε στάνταρ τιμές (0.8 και 0.2 αντίστοιχα) και θα βελτιστοποιηθούν παρακάτω.</p>
# <p>Όσον αφορά τους γενετικούς τελεστές, επελέγησαν κάποιοι από το σύνολο των συμβατών με float αριθμούς. Και για εξελικτικούς αλγόριθμους επιλέξαμε τους 3 δοσμένους (απλό/ μ,λ/ μ+λ).</p>
# <p>Είμαστε έτοιμοι λοιπόν για την τελική αναζήτηση.</p> 

# In[60]:


evalFunc = mishra9


# In[61]:


print "max time = " + str(stats_table['avg.time'].max())
print "min time = " + str(stats_table['avg.time'].min())


# In[62]:


stats_table = multipleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM)


# #### Οπτικοποίηση αποτελεσμάτων
# <p>Παρατηρούμε ότι για αρκετούς συνδυασμούς τελεστών - στρατηγικών δεν υπήρξε καμία επιτυχία εξαιτίας του σχετικά μεγάλου delta που επιλέχθηκε.</p>

# In[65]:


stats_table


# Θα βρούμε τώρα τους 5 καλύτερους συνδυασμούς τελεστών - στρατηγικών με βάση τη μετρική avg.min, και θα εξετάσουμε σε πόσους γύρους βρήκαν επιτυχώς ελάχιστη τιμή μικρότερη του DELTA (υπενθυμίζουμε ότι GOAL=0).

# In[84]:


stats_table.iloc[stats_table['avg.min'].argsort()[0:5]]


# <p>Όπως φαίνεται από πάνω, οι καλύτεροι συνδυασμοί είναι αυτοί με id 69, 65, 63, 57 και 53.</p>
# <p>Θα αποφανθούμε τώρα ως προς το ποιον θα κρατήσουμε εως καλύτερο.</p>
# <p>Αρχικά ο 53 δεν βρίσκει επιτυχώς τιμή μικρότερη του DELTA, οπότε τον αποκλείουμε.</p>
# <p>Από τους υπόλοιπους βλέπουμε ότι οι 65, 63 και 57 συγκλίνουν γρηγορότερα σε τιμή μικρότερη του DELTA (45+ γενεές) σε σχέση με τον 69 (60 γενεές).</p>
# <p>Παρ' όλ' αυτά ο 69 έχει καλύτερο avg.min απο αυτούς κατά 3 τάξεις μεγέθους.</p>
# <p>Καταλαβαίνουμε λοιπόν ότι οι 65,63 και 57 συγκλίνουν λίγο γρηγορότερα, όμως ο 69 συνεχίζει να βρίσκει ολοένα και καλύτερες λύσεις ενόσω οι υπόλοιποι μένουν πιο στάσιμοι.</p>
# <p>Καταλήγουμε λοιπόν να θεωρούμε τον 69 ως τον καλύτερο συνδυασμό.</p>

# In[92]:


print "Best combination is the one with id 69, having.."
print "Operators: " + str(stats_table['operators'][69])
print "And strategy: " + str(stats_table['strategy'][69])


# ### Τελική βελτιστοποίηση

# #### Βελτιστοποίηση πιθανοτήτων διασταύρωσης και μετάλλαξης

# <p>Ο καλύτερος συνδυασμός τελεστών - στρατηγικής, όπως εξηγήθηκε παραπάνω, είναι ο εξής:<br>
#     \(\bullet\) Operators: cxSimulatedBinaryHigh,mutShuffleIndexesLow,SelectionLow<br>
#     \(\bullet\) Strategy: eaSimple 280 0.8 0.2</p>
# <p>Θα προσπαθήσουμε τώρα να βρούμε το βέλτιστο συνδυασμό πιθανοτήτων διασταύρωσης και μετάλλαξης κάνοντας grid search στο διάστημα [0.05, 0.9]</p>
# <p>Για αρχή θα ψάξουμε στις τιμές 0.05, 0.1, 0.2, 0.3,.., 0.9 και εν συνεχεία σε πιο μικρά διαστήματα για μεγαλύτερη ακρίβεια.</p>

# In[103]:


# best operators - strategy combo
cxAlgoM = [tools.cxSimulatedBinary]
cxArgsM = [{'eta':1}]
mutAlgoM = [tools.mutShuffleIndexes]
mutArgsM = [{'indpb':0.7/numVariables}]
selArgsM = [{'tournsize':10}]
eaAlgoM = [algorithms.eaSimple]
eaArgsM = [{}]


# In[107]:


# grid search on probabilities
pb = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cxpbM = pb
mutpbM = pb
stats_table2 = multipleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM)


# Ελέγχουμε πάλι τους 5 καλύτερους συνδυασμούς πιθανοτήτων με βάση το avg.min.

# In[110]:


stats_table2.iloc[stats_table2['avg.min'].argsort()[0:5]]


# Θα κρατήσουμε τον καλύτερο εξ' αυτών και θα κάνουμε ένα ακόμα grid search σε μικρότερο διάστημα.

# In[111]:


# grid search on probabilities in shorter interval
cxpbM = [0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94]
mutpbM = [0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64]
stats_table3 = multipleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM)


# Ελέγχουμε για ακόμα μία φορά τους 5 καλύτερους συνδυασμούς πιθανοτήτων με βάση το avg.min.

# In[112]:


stats_table3.iloc[stats_table3['avg.min'].argsort()[0:5]]


# <p>Οι τελικές τιμές που θα κρατήσουμε επομένως είναι:</p>
# <p>\(\bullet\) Πιθανότητα διασταύρωσης: 0.92</p>
# <p>\(\bullet\) Πιθανότητα μετάλλαξης: 0.64</p>

# #### Εύρεση βέλτιστης (ελάχιστης) τιμής της συνάρτησης με τον ΓΑ
# <p>Θα κάνουμε ένα run του τελικού αλγορίθμου που προέκυψε με μεγάλο αριθμό γενεών και πληθυσμού ώστε να πάρουμε μία βέλτιστη τιμή.</p>

# In[55]:


# best operators - strategy combo
cxAlgo = tools.cxSimulatedBinary
cxArgs = {'eta':1}
mutAlgo = tools.mutShuffleIndexes
mutArgs = {'indpb':0.7/numVariables}
selArgs = {'tournsize':10}
eaAlgo = algorithms.eaSimple
eaArgs = {}
# best probabilities
cxpb = 0.92
mutpb = 0.64
# other
evalFunc = mishra9
numVariables = 3
gens = 200
popul = 500
lg, xtime, hf = ea_func(gens,numVariables, evalFunc, cxAlgo, cxArgs, mutAlgo, mutArgs, selArgs, eaAlgo, eaArgs, popul, cxpb, mutpb)


# In[56]:


best_indiv = hf[0]
best_value = hf[0].fitness
total_evals = sum(lg.select("nevals"))
print "Best individual is: " + str(best_indiv) + " with fitness value: " + str(best_value)
print "Total evaluations: " + str(total_evals)
print "Execution time: " + str(xtime)


# Βρήκαμε επομένως ένα ολικό ελάχιστο (βέλτιστο) της συνάρτησης, καθότι ξέρουμε ότι η ελάχιστη τιμή που μπορεί να δώσει είναι 0.

# ## Μέρος 2. Μελέτη κλιμακούμενης συνάρτησης
# <br><center><b>Schwefel 2.22 Function</b></center><br>
# <p>$$f(\mathbf{x})=f(x_1, ..., x_n)=\sum_{i=1}^{n}|x_i|+\prod_{i=1}^{n}|x_i|$$</p>
# <p>\(\bullet\) $f(\textbf{x}^{\ast})=0$ at $\textbf{x}^{\ast} = (0, …, 0)$.</p>
# <p>\(\bullet\) $x_i \in [-100, 100]$ for $i=1, …, n$.</p>

# ### Για D=2 
# - α) Εκτύπωση "3D" γραφήματος της συνάρτησης $f(x1,x2)$ και περιγραφή της μορφής της.

# In[152]:


# Schwefel 2.22 Function

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
x1 = x2 = np.linspace(-100, 100, 2000)
X1, X2 = np.meshgrid(x1, x2)
ys = np.array([schwefel2_22([x1,x2])[0] for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])
Y = ys.reshape(X1.shape)

ax.plot_surface(X1, X2, Y)

ax.set_xlabel('X1 Label')
ax.set_ylabel('X2 Label')
ax.set_zlabel('Y Label')

plt.show()


# $\Rightarrow$ Η συνάρτηση φαίνεται να έχει ολικό ελάχιστο για $(x_1,x_2)=(0,0)$ όπως άλλωστε γνωρίζαμε ήδη.
# <br>$\Rightarrow$ Τα ολικά της μέγιστα σχηματίζονται στα σημεία:<br> 
#  $(x_1,x_2)=(-100,-100)$<br>$(x_1,x_2)=(-100,100)$<br>$(x_1,x_2)=(100,-100)$<br>$(x_1,x_2)=(100,100)$.
# <br>$\Rightarrow$ Φαίνεται να χωρίζεται σε 4 κοίλα συνεχή τμήματα, μεταξύ των οποίων τα όρια ασυνέχειας βρίσκονται στην τομή του γραφήματος με τα επίπεδα $x_1=0$ και $x_2=0$.
# <br>$\Rightarrow$ Κάθε ένα από αυτά τα 4 τμήματα είναι γνησίως αύξον (αντίστοιχα γνησίως φθίνον) ως προς $x_i$ αν ορίζεται στις θετικές τιμές (αντίστοιχα στις αρνητικές τιμές) του $x_i$. Η σχέση αυτή ισχύει και για τη συνάρτηση την ίδια γενικότερα.

# - β) Εύρεση βέλτιστου γενετικού αλγορίθμου και βέλτιστης τιμής για το πρόβλημα.
# <p>Θα κινηθούμε όπως και πριν, βρίσκοντας αρχικά τον βέλτιστο συνδυασμό τελεστών - στρατηγικής.</p>

# In[175]:


# find limit so that only 10 combinations have avg.min below DELTA
sum(stat_table['avg.min'] < 6e-70)
# execution has been done in advance from cell down below


# In[6]:


# function parameters
numVariables = 2
evalFunc = schwefel2_22
# operators
cxAlgoM = [tools.cxBlend, tools.cxSimulatedBinary]
cxArgsM = [{'alpha':0.7}, {'alpha':0.3}, {'eta':1}, {'eta':5}]
mutAlgoM = [tools.mutGaussian, tools.mutShuffleIndexes]
mutArgsM = [{'mu':0, 'sigma':5, 'indpb':1./numVariables}, {'mu':0, 'sigma':1, 'indpb':1./numVariables}, {'indpb':1.3/numVariables}, {'indpb':0.7/numVariables}]
selArgsM = [{'tournsize':2}, {'tournsize':10}]
# strategy
eaAlgoM = [algorithms.eaSimple, algorithms.eaMuPlusLambda, algorithms.eaMuCommaLambda]
popul = 280 # population
eaArgsM = [{}, {'mu':popul*1/4, 'lambda_':popul*3/4}, {'mu':popul*1/4, 'lambda_':popul*3/4}]
cxpbM = [0.8]
mutpbM = [0.2]
# other
rounds = 10
gens = 150
delta = 6e-70

stat_table = multipleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM)


# Εμφάνιση 5 καλύτερων επιλογών με βάση το avg.min:

# In[7]:


stat_table.iloc[stat_table['avg.min'].argsort()[0:5]]


# Ο συνδυασμός που επιλέχθηκε ως βέλτιστος είναι:

# In[8]:


print "Best combination is the one with id 45, having.."
print "Operators: " + str(stat_table['operators'][45])
print "And strategy: " + str(stat_table['strategy'][45])


# Θα κάνουμε τώρα grid search στις πιθανότητες:

# In[12]:


# best operators - strategy combo
cxAlgoM = [tools.cxBlend]
cxArgsM = [{'alpha':0.3}]
mutAlgoM = [tools.mutShuffleIndexes]
mutArgsM = [{'indpb':0.7/numVariables}]
selArgsM = [{'tournsize':10}]
eaAlgoM = [algorithms.eaSimple]
eaArgsM = [{}]


# In[13]:


# grid search on probabilities
pb = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cxpbM = pb
mutpbM = pb
stat_table2 = multipleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM)


# Εμφάνιση 5 καλύτερων επιλογών με βάση το avg.min:

# In[14]:


stat_table2.iloc[stat_table2['avg.min'].argsort()[0:5]]


# Grid search στις πιθανότητες σε μικρό διάστημα γύρω από τις βέλτιστες τιμές:

# In[15]:


# grid search on probabilities in shorter interval
cxpbM = [0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94]
mutpbM = [0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64]
stat_table3 = multipleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM)


# Εμφάνιση 5 καλύτερων επιλογών με βάση το avg.min:

# In[16]:


stat_table3.iloc[stat_table3['avg.min'].argsort()[0:5]]


# Επομένως οι βέλτιστες πιθανότητες που θα επιλέξουμε είναι 0.93 και 0.57 για διασταύρωση και μετάλλαξη αντίστοιχα.

# Τελική εκτέλεση του αλγορίθμου με τις πλέον δεδομένες παραμέτρους για να βρούμε το βέλτιστο χρωμόσωμα, την αντίστοιχη fitness value και μαζί και τις συνολικές αποτιμήσεις και το χρόνο εκτέλεσης.

# In[41]:


# best operators - strategy combo
cxAlgo = tools.cxBlend
cxArgs = {'alpha':0.3}
mutAlgo = tools.mutShuffleIndexes
mutArgs = {'indpb':0.7/numVariables}
selArgs = {'tournsize':10}
eaAlgo = algorithms.eaSimple
eaArgs = {}
# best probabilities
cxpb = 0.93
mutpb = 0.57
# other
evalFunc = schwefel2_22
numVariables = 2
gens = 600
popul = 500
lg2, xtime2, hf2 = ea_func(gens,numVariables, evalFunc, cxAlgo, cxArgs, mutAlgo, mutArgs, selArgs, eaAlgo, eaArgs, popul, cxpb, mutpb)


# In[42]:


best_indiv = hf2[0]
best_value = hf2[0].fitness
total_evals = sum(lg2.select("nevals"))
print "Best individual is: " + str(best_indiv) + " with fitness value: " + str(best_value)
print "Total evaluations: " + str(total_evals)
print "Execution time: " + str(xtime2)


# Και βρήκαμε τελικά το ολικό ελάχιστο στο σημείο $(0,0)$.

# ### Για D=1, 10, 20, 40 και μεγαλύτερες διαστάσεις

# Θα ορίσουμε συνάρτηση που βασίζεται στην multipleEaStats για την δημιουργία πίνακα (για συγκεκριμένο αλγόριθμο, εδώ τον βέλτιστο που βρέθηκε παραπάνω) με: **αριθμό διαστάσεων**, **αριθμό επιτυχιών**, **μέσο ολικό ελάχιστο**, **μέσο αριθμό αποτιμήσεων** και **μέσο χρόνο**.
# <p>Θα δέχεται τα ίδια ορίσματα με την multipleEaStats, όμως το numVariables θα λέγεται πλέον numVariablesM και θα είναι διάνυσμα από πλήθος μεταβλητών.</p>
# <p>Κατ' αντιστοιχία θα αλλάξει και το delta σε deltaM.</p>
# <p>* Επειδή ο τελεστής μετάλλαξης mutShuffleIndexes που επιλέχτηκε στο βέλτιστο συνδυασμό απαιτεί τουλάχιστον δύο γονίδια ανά χρωμόσωμα ώστε να κάνει shuffle, αντί για D=1 θα κρατήσουμε το D=2 να αντιπροσωπεύει τη συγκεκριμένη τάξη μεγέθους.</p> 

# In[71]:


# function to return DataFrame with headers as below for given algorithm
def variousDimStats(deltaM, rounds, gens, numVariablesM, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM):
    headers = ['dimensions', 'successes', 'avg.min', 'avg.evals', 'avg.time']
    table = []
    for numVariables in numVariablesM:
        delta = deltaM[numVariablesM.index(numVariables)]
        df = multipleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM)
        row = [numVariables]
        row.append(df['successes'][0])
        row.append(df['avg.min'][0])
        row.append(df['avg.evals'][0])
        row.append(df['avg.time'][0])
        table.append(row)
    tableDF = pd.DataFrame(table, columns=headers)
    return tableDF


# Θα αυξήσουμε τους γύρους σε 15, και πάμε να τρέξουμε για το βέλτιστο αλγόριθμο.

# In[74]:


# function parameters
numVariablesM = [2, 10, 20, 40, 80, 160]
evalFunc = schwefel2_22
# operators
cxAlgoM = [tools.cxBlend]
cxArgsM = [{'alpha':0.3}]
mutAlgoM = [tools.mutShuffleIndexes]
mutArgsM = [{'indpb':0.7/numVariables}]
selArgsM = [{'tournsize':10}]
# strategy
eaAlgoM = [algorithms.eaSimple]
popul = 280 # population
eaArgsM = [{}]
cxpbM = [0.93]
mutpbM = [0.57]
# other
rounds = 15
gens = 150
deltaM = [3e-80, 3e-28, 3e-19, 4e-13, 7e-8, 6e-4]
statTable = variousDimStats(deltaM, rounds, gens, numVariablesM, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM)


# In[75]:


statTable


# α) Σχολιασμός αποτελεσμάτων του πίνακα:
# <p>Παρατηρούμε ότι όσο αυξάνονται οι διαστάσεις αυξάνεται και το avg.min (κατά τάξεις μεγέθους), γι' αυτό άλλωστε χρησιμοποιήσαμε και διαφορετικό DELTA για κάθε αριθμό διαστάσεων.</p>
# <p>Οι μέσες αποτιμήσεις παραμένουν σταθερές, πράγμα απολύτως φυσιολογικό αφού οι γενεές, ο πληθυσμός και η γενική στρατηγική μας είναι κοινές για όλα τα διαφορετικά μεγέθη διαστάσεων.</p>
# <p>Ο μέσος χρόνος παρ' όλα αυτά αυξάνεται. Αυτό συμβαίνει διότι κάθε αποτίμηση είναι πιο χρονοβόρα όσο αυξάνονται οι διαστάσεις, μιας και θα πρέπει η συνάρτηση να προσπελάσει πιο πολλά γονίδια ώστε να υπολογίσει το αποτέλεσμα.</p>

# β) Οι αιτίες του φαινομένου:
# <p>Προφανώς ο υπαίτιος στην περίπτωση μας για την αύξηση του avg.min είναι το μέγεθος των διαστάσεων, αφού είναι ο μοναδικός παράγοντας που αλλάζει.</p>
# <p>Όσο μεγαλώνει η διαστατικότητα, τόσο πιο δύσκολο είναι να βρεθούν καλά σημεία που να δίνουν τιμές κοντά στο 0, μιας και οι δυνατοί συνδυασμοί γονιδίων αυξάνονται εκθετικά.</p>
# <p>Μεγαλώνει δηλαδή ο χώρος αναζήτησης, οι γενεές όμως (και οι συνολικές αποτιμήσεις εν γένει) παραμένουν σταθερές.</p>
# <p>Μία ακόμα αιτία είναι ότι στη συγκεκριμένη συνάρτηση (Schwefel 2.22) ο ένας από τους δύο κύριους όρους είναι το άθροισμα των απολύτων τιμών των γονιδίων. Ο όρος αυτός είναι ο κύριος παράγοντας για το αποτέλεσμα που θα μας δώσει η συνάρτηση όταν έχουμε τιμές γονιδίων μικρότερες της μονάδας. Επομένως είναι φυσικό όσο αυξάνεται το πλήθος των γονιδίων (διαστάσεις) να μας επιστρέφονται ολοένα και χειρότερες (μεγαλύτερες) τιμές αφού αθροίζονται περισσότερα γονίδια.</p>

# γ) Τρόποι βελτίωσης αποτελεσμάτων σε μεγάλες διαστάσεις:<br><br>
# Η μία προφανής λύση θα ήταν η αύξηση των γενεών. Υποθέτοντας σταθερό αριθμό γενεών οι δυνατές λύσεις είναι οι εξής:
# - Αύξηση του αρχικού πληθυσμού ώστε να μεγαλώσει το εύρος των διαθέσιμων σημείων και να ισορροπήσει με την αύξηση του χώρου αναζήτησης.
# - Μία πιο χρονοβόρα προσέγγιση θα ήταν να ψάξουμε από την αρχή το βέλτιστο συνδυασμό γενετικών τελεστών - στρατηγικών εξέλιξης και πιθανοτήτων. Αυτό βέβαια θα παραήταν σπάταλο και δεν το στηρίζουμε σε κάποια σαφή τεκμηρίωση, παρά μόνο στη διαίσθηση ότι ίσως ο συνδυασμός που ακολουθούμε δεν είναι βέλτιστος για όλες τις διαστάσεις.

# ### Βελτιστοποίηση σε μεγάλες διαστάσεις
# Θα εφαρμόσουμε μεταβολή πληθυσμού (όπως αναφέρθηκε στο γ) για τους ακόλουθους δύο στόχους:

# 1. Κρατώντας σταθερή και μεγάλη διάσταση εισόδου να παίρνουμε όλο και καλύτερες βέλτιστες τιμές της συνάρτησης.
# 2. Για δύο διαδοχικές μειώσεις κατά ήμισυ του DELTA να παίρνουμε κάθε φορά μεγαλύτερο ποσοστό επιτυχιών (την πρώτη φορά διπλάσιο και τη δεύτερη φορά απλά να αυξηθεί).

# Ορίζουμε αρχικά μία συνάρτηση παρόμοια με την προηγούμενη, στην οποία όμως θα έχουμε στήλη με αριθμό πληθυσμού αντί για διαστάσεων.

# In[76]:


def variousPopStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, populM, cxpbM, mutpbM):
    headers = ['population', 'successes', 'avg.min', 'avg.evals', 'avg.time']
    table = []
    for popul in populM:
        df = multipleEaStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, popul, cxpbM, mutpbM)
        row = [popul]
        row.append(df['successes'][0])
        row.append(df['avg.min'][0])
        row.append(df['avg.evals'][0])
        row.append(df['avg.time'][0])
        table.append(row)
    tableDF = pd.DataFrame(table, columns=headers)
    return tableDF


# 1. Ξεκινάμε από τον πρώτο στόχο.<br>
# <p>Επιλέγουμε διάσταση εισόδου 100</p>
# <p>Για τιμές πληθυσμού θα χρησιμοποιήσουμε ως ελάχιστη μία κοντά σε αυτήν που χρησιμοποιήθηκε προηγουμένως και θα αυξάνουμε ανά 50.</p>

# In[77]:


# function parameters
numVariables = 100
evalFunc = schwefel2_22
# operators
cxAlgoM = [tools.cxBlend]
cxArgsM = [{'alpha':0.3}]
mutAlgoM = [tools.mutShuffleIndexes]
mutArgsM = [{'indpb':0.7/numVariables}]
selArgsM = [{'tournsize':10}]
# strategy
eaAlgoM = [algorithms.eaSimple]
populM = [300, 400, 500, 600, 700, 800, 900, 1000] # population
eaArgsM = [{}]
cxpbM = [0.93]
mutpbM = [0.57]
# other
rounds = 15
gens = 150
delta = 1e-4
statTable2 = variousPopStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, populM, cxpbM, mutpbM)


# In[78]:


statTable2


# Παρατηρούμε ότι οι τιμές του avg.min σταδιακά μειώνονται όσο αυξάνεται ο πληθυσμός. Άρα όντως ο παράγοντας πληθυσμός μπορεί να βελτιώσει τα αποτελέσματα όταν έχουμε σταθερές γενεές αλλά όχι και δραματικά.

# 2. Για αυτό το στόχο θα χρησιμοποιήσουμε διάσταση εισόδου 50.
# <p>Πάμε αρχικά να βρούμε ένα DELTA που να μας δίνει ποσοστό επιτυχιών γύρω στο 35% - 50%. Δηλαδή κάπου 5 με 7 επιτυχίες, δεδομένου ότι έχουμε θέσει 15 γύρους.</p>

# In[86]:


numVariables = 50
populM = [300]
delta = 3.5e-7
statTable3 = variousPopStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, populM, cxpbM, mutpbM)


# In[87]:


statTable3


# Βρήκαμε ένα DELTA για το οποίο έχουμε 6 επιτυχίες. Μειώνουμε τώρα το DELTA στο μισό από αυτό που βρήκαμε και ελέγχουμε αν μπορούμε να διπλασιάσουμε τις επιτυχίες.

# In[88]:


delta /= 2
populM = [300, 500, 700, 900, 1100, 1300]
statTable4 = variousPopStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, populM, cxpbM, mutpbM)


# In[89]:


statTable4


# Οι επιτυχίες όχι μόνο διπλασιάστηκαν με την αύξηση του πληθυσμού, αλλά έφτασαν μάλιστα στο μέγιστο δυνατό. Μειώνουμε εκ νέου το DELTA κατά το ήμισυ και ελέγχουμε αν μπορούμε να αυξήσουμε και πάλι το ποσοστό επιτυχιών. Είναι βέβαια προφανές και από τον παραπάνω πίνακα ότι το avg.min πέφτει μέχρι και 4 τάξεις μεγέθους, επομένως περιμένουμε να έχουμε και πάλι ποσοστό επιτυχιών 100%.

# In[90]:


delta /= 2
populM = [300, 500, 700, 900, 1100, 1300]
statTable5 = variousPopStats(delta, rounds, gens, numVariables, evalFunc, cxAlgoM, cxArgsM, mutAlgoM, mutArgsM, selArgsM, eaAlgoM, eaArgsM, populM, cxpbM, mutpbM)


# In[91]:


statTable5


# Άρα καταφέραμε και πάλι να αυξήσουμε το ποσοστό των επιτυχιών στο 100% παρά την εκ νέου μείωση του DELTA, η οποία έριξε το ποσοστό των επιτυχιών στο 0% για τον αρχικό πληθυσμό.
