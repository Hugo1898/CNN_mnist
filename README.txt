
Ce projet contient un main qui peut être executé à partir de la commande : make_run_accur

#####################################################################################################################

make_run_accur : utilise le main du fichier "main_stochastic_grad_accur.c" et permet d'afficher dans la console l'évolution de la 
fonction coût en sortie du réseau de neuronne, ainsi que la précision de ce réseau à chaque itération. Ce code s'arrête dès que la 
précision voulu "accur_seek est atteinte ou que le nombre max d'itération "nb_iter_max" est atteint.
Un fichier csv est également généré dans ./data/ et contient toutes les valeurs prise par la fonction coût et la précision, uniquement
quand la précision augmente (càd sans le bruit des oscillations pour faciliter la lecture des données).

#########################################################################################################################

Au début du main, deux sections contiennent des paramètres que l'on peut modifier.

CNN size and characteristics contient :

int base_size_r = 32; //number of rows of the input
int base_size_c = 32; //number of columns of the input
int kernel_size = 3; // change PAD const if you change kernel_size !! (cf kernel.h)
int padding = 1; // activation of padding or not. (padding = 0 padding deactivated, padding = 1 padding activated)
int pool = 1; // pool == 0 for max_pool and pool == 1 for avg_pool
static int FC_sizes[NB_FC_LAYERS+1] = {0, 15, 10, 2};

Lors de la modification de ces données il faut faire attention à être cohérent avec les constantes présentes dans "nn.h" et "kernel.h".

kernel_size correspond à la taille du kernel, ici 3x3 et doit donc touours être impaire !

PAD dans kernel.h doit toujours être égal à kernel_size - 1.

FC_SIZES doit toujours contenir NB_FC_LAYERS + 1 éléments. En  commençant par 0 et finissant par le nombre de output du Fully connected (ici 2 
car on a que deux labels qui sont matrice horizontale ou verticale). Donc ici par exemple pour static int FC_sizes[NB_FC_LAYERS+1] = {0, 15, 10, 2};
avec NB_FC_LAYERS = 3 on aura un FC avec 3 layers composées respectivement de 15, 10 et 2 neuronnes.

---------------------------------------------------------------------------

Learning and test variables contient :

float lr = 0.2f; // learning rate
int nb_iter_max = 2500;
int nb_test = 250;
float seek_accur = 0.93f; // accur sought
int size_avg = 5; // batch_size

nb_iter_max correspond au nombre d'itéraction max qu'on autorise pour entrainer le réseau de neuronnes.

nb_test correspond au nombre d'évaluation qu'on veut pour obtenir la précision du réseau de neuronnes.

size_avg définit la taille du batch qu'on utilise à chaque itération pour entraîner le réseau de neuronnes.

"seek_accur" est la précision qu'on souhaite atteindre.

------------------------------------------------------------------------------

Dans le fichier "kernel.h" :

#define MAX_F_ROWS 10 // kernel_size_max
#define MAX_F_COLUMNS 10 // kernel_size_max
#define PAD 2 //kernel_size - 1
#define POOL_SIZE 2 // Pooling : POOL_SIZExPOOL_SIZE 

POOL_SIZE correspond à la taille du pooling et doit donc toujours être paire (ici le pooling est de taille 2x2).

------------------------------------------------------------------------------

Dans le fichier "nn.h" :

#define NB_LAYERS 3 // Number of CNN layers
#define NB_FC_LAYERS 3 // Number of FC layers
#define NOM 2 // Type of activation function used : RELU == 0, SIGMOID == 1; TANH == 2
#define COST 0 // Type of cost function used : MSE == 0, BCE == 1

----------------------------------------------------------------------------

Dans le fichier "backprop_runner.h" :

#define BETA 1.0f // decreases the learning rate at each iteration if != 1.0f

BETA permet de diminuer le learning rate à chaque itération si sa valeur est différente de 1, en général on prend
BETA = 0.9999

!!! Une attention particulière devra être apporté lors du choix de kernel_size, POOL_SIZE et NB_LAYERS pour s'assurer qu'à la sortie de la couche de CNN,
en fonction de l'activation ou non du padding, on a bien une sortie qui ne soit pas vide car l'image a été trop réduite !!!


