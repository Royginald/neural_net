

%% ------------------------ intial decleration ------------------------ %%
if strcmp(input("New weights (n) or stored weights (s)? ", 's'), "n")
    if strcmp(input("Use default size (4 layers, 3 nodes, 5 outputs, 3 inputs)? (y/n) ", 's'), "y")
        Nlayers = 4;
        Nnodes = 3;
        Noutputs = 5;
        Ninputs = 3;
    else
        Nlayers = input("Number of layers? ");
        Nnodes = input("Number of nodes in each layer? ");
        Noutputs = input("Number of outputs? ");
        Ninputs = input("Number of inputs? ");
    end

    weights = cell(Nlayers+2, 1);
    weights{1} = rand([Nnodes, Ninputs]);
    for i = 2:Nlayers + 1
        weights{i} = rand([Nnodes, Nnodes]);
    end
    weights{Nlayers+2} = rand([Noutputs, Nnodes]);
else
    Nlayers = length(weights)-2;
    Nnodes = length(weights{1});
    Noutputs = length(weights{end});
    Ninputs = length(weights{1}(1, :));
end

inputs = zeros(Ninputs, 1);
layers = cell(Nlayers+3, 1);

disp("Weights:");
for i = 1:length(weights)
   disp(weights{i});
end

%% ---------------------------- test input ---------------------------- %%
if strcmp(input("Compute for an input? (y/n) ", 's'), "y")
    if strcmp(input("Manual entry (m) or read from file (r)? ", 's'), "m")
        disp("Enter inputs:");
        for i = 1:Ninputs
            inputs(i) = input("Input number " + i + ": ");
        end
    else
        if Ninputs > 10
            disp("Good choice");
        end

        entryNumber = input("Enter the index of the number you would like to view: ");

        data = readtable('test.csv');

        inputs = data{:, :}(entryNumber, :)';
    end
    
    layers{1} = inputs;
    for i = 1:Nlayers+2
        layers{i+1} = ReLu(weights{i} * layers{i});
    end
    
    disp("Output:");
    disp(layers{end});
end

%% ------------------------------ Train ------------------------------- %%
if strcmp(input("Train? (y/n) ", 's'), "y")

    % need 784 inputs and 10 outputs for this training set
    training_data = readtable('train.csv');
    training_data = training_data{:, :};

    training = cell(height(training_data), 2);

    for i = 1:length(training)
        training{i, 1} = training_data(i, 2:end)';
        training{i, 2} = zeros(10, 1);
        training{i, 2}(training_data(i, 1)+1) = 1;
    end

%     training = cell(10, 2);
%     for i = 1:length(training)
%         training{i, 1} = rand([Ninputs, 1]);
%         training{i, 2} = rand([Noutputs, 1]);
%     end
    
    costs = cell(length(training), 1);
    errors = cell(length(training), length(layers));
    gradWeight = cell(length(training), length(weights));

    for i = 1:length(training)
        layers{1} = training{i, 1};
        for j = 1:Nlayers+2
            layers{j+1} = ReLu(weights{j} * layers{j});
        end
        costs{i} = training{i, 2} - layers{end};

        errors{i, end} = step(layers{end}) .* costs{i};

        for j = length(layers)-1:-1:1
            errors{i, j} = Hadamard(step(layers{j}), weights{j}') * errors{i, j+1};                   
        end

        for j = 1:length(weights)
            gradWeight{i, j} = errors{i, j+1} * layers{j}';
        end
    end

    scale = 0.001;

    for i = 1:length(training)
        for j = 1:length(weights)
            weights{j} = weights{j} + gradWeight{i, j} * scale;
        end
    end
end

%% ---------------------------- Functions ----------------------------- %%

function mOut = ReLu(mIn)
    sz = size(mIn);
    mOut = zeros(sz);

    for i = 1:sz(1)
        for j = 1:sz(2)
            mOut(i, j) = max(0, mIn(i, j));
        end
    end
end

function mOut = step(mIn)
    sz = size(mIn);
    mOut = zeros(sz);

    for i = 1:sz(1)
        for j = 1:sz(2)
            if mIn(i, j) > 0
                mOut(i, j) = 1;
            else
                mOut(i, j) = 0;
            end
        end
    end
end

function mOut = Hadamard(m1, m2)
    sz = size(m2);
    mOut = zeros(sz);
    for i = 1:sz(1)
        for j = 1:sz(2)
            mOut(i, j) = m1(i) * m2(i, j);
        end
    end
end
    

% https://en.wikipedia.org/wiki/Backpropagation
% https://en.wikipedia.org/wiki/Activation_function
% https://en.wikipedia.org/wiki/Matrix_calculus
