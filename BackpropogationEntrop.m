%Shivam Swarnkar (ss8464)
%BP algorithm for one hidden layer NN (N0-N1-N2) using cross entropy 
function result = BackpropogationEntrop(X,Y,n0,n1, alpha, range, xO)
    
    epochs = 50000;
    T = 0.05;
    n2 = 1;
    n = Init(n0, n1, n2,range);
    n = nnTrain(n, X,Y, epochs, alpha,T,xO);
    n.error = err(n, X,Y,xO);
    result = n;
end


%initialize neural network
function nn = Init(n0, n1, n2,range)
 
   
    nn.wH = -range + ((2*range) * rand(n0, n1));
    nn.bH = -range + ((2*range) *rand(1, n1));
    
    nn.wO = -range + ((2*range) *rand(n1, n2));
    nn.bO = -range + ((2*range) *rand(1, n2));

    nn.Loss = [];
    nn.Accuracy = [];
    
    %for testing, comment out later and use rand
%     nn.wH = [0.1970 0.3191 -0.1448 0.3594; 0.3099 0.1904 -0.0347 -0.4861];
%     nn.bH = [-0.3378 0.2771 0.2859 -0.3329];
%     
%     nn.wO = [0.4919 -0.2913 -0.3979 0.3581]';
%     nn.bO = -0.1401;
%   
end


%train neural network with BP
function nn = nnTrain(nn, X, Y, epochs, alpha, T,xO)
    i = 1;
    curr_err = 100; %so that first epoch can run
    while (i<=epochs && curr_err >= T)
        for j = 1:size(X,1)
            nn = nnEval(nn, X(j, :), xO);
           
            %s(L)  Cross Entropy 
            delta =( -2/((nn.o + Y(j,:))))*sigmoid_grad(nn.aO, xO);
            
            %calculating s(l), and change value
            %--------------------------%
            wO_cor = alpha * delta * nn.zH';
            bO_cor = alpha * delta;
            delta_H = (sigmoid_grad(nn.aH, xO)) .* (nn.wO' *delta) ;
            wH_cor = alpha * X(j,:)' * delta_H;
            bH_cor = alpha * delta_H;
            %--------------------------%
            
            %update weights and bias
            
            nn.wH = nn.wH-wH_cor;
            nn.bH = nn.bH - bH_cor;
            
            nn.wO = nn.wO-wO_cor;
            nn.bO = nn.bO -bO_cor;
            
            
        end
        i = i+1;
        curr_err = err(nn,X,Y,xO);
       
    end

    if(i<epochs)
        nn.converged = true;
        nn.epochs = i;
    else
        nn.converged=false;
    end
end


%feed forward
function nn = nnEval(nn, X,xO)
    nn.aH = X * nn.wH + nn.bH;
    nn.zH = sigmoid(nn.aH,xO);
    nn.aO =  nn.zH *nn.wO+ nn.bO;
    nn.o = sigmoid(nn.aO,xO);
end

%transfer function
function val = sigmoid(val,xO)
    val = val/xO;
    val = (1-exp(-val))./(1+exp(-val));
    
end

%gradient transfer function
function val = sigmoid_grad(val,xO)
    sig_val = sigmoid(val,xO);
    val =  (1- (sig_val.^2))/(xO*2);
end

%finds total error in prediction
function result = err(n,X,Y,xO)
    nn = nnEval(n, X,xO);
    result = sum((nn.o-Y).^2);
end
