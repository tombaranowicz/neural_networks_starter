import Accelerate

private var inputFilter: BNNSFilter?
private var outputFilter: BNNSFilter?

let inputWeights: [Float] = [ -6.69970608,  5.47385073, 6.81709385, -5.15285587 ]
let inputBiases: [Float] = [ 3.46804953,  2.56963539]
let outputWeights: [Float] = [-6.58759022, -6.7153616 ]
let outputBiases: [Float] = [ 9.69208241 ]

func testNetwork(_ input: [Float]) {
    
    var hidden: [Float] = [0, 0]
    var output: [Float] = [0]
    
    //@return 0 on success, and -1 on failure.
    if BNNSFilterApply(inputFilter, input, &hidden) != 0 {
        print("Hidden Layer failed.")
        return
    }
    
    if BNNSFilterApply(outputFilter, hidden, &output) != 0 {
        print("Output Layer failed.")
        return
    }
    
    print("Testing [\(input)] = \(output[0])")
}

func test() {
    let activation = BNNSActivation(function: BNNSActivationFunctionSigmoid, alpha: 0, beta: 0)
    
    //1st filter
    let inputToHiddenWeightsData = BNNSLayerData(
        data: inputWeights, data_type: BNNSDataTypeFloat32,
        data_scale: 0, data_bias: 0, data_table: nil)
    
    let inputToHiddenBiasData = BNNSLayerData(
        data: inputBiases, data_type: BNNSDataTypeFloat32,
        data_scale: 0, data_bias: 0, data_table: nil)
    
    var inputToHiddenParams = BNNSFullyConnectedLayerParameters(
        in_size: 2, out_size: 2, weights: inputToHiddenWeightsData,
        bias: inputToHiddenBiasData, activation: activation)
    
    var inputDescriptor = BNNSVectorDescriptor(
        size: 2, data_type: BNNSDataTypeFloat32, data_scale: 0, data_bias: 0)
    
    var hiddenDescriptor = BNNSVectorDescriptor(
        size: 2, data_type: BNNSDataTypeFloat32, data_scale: 0, data_bias: 0)
    
    inputFilter = BNNSFilterCreateFullyConnectedLayer(&inputDescriptor, &hiddenDescriptor, &inputToHiddenParams, nil)
    guard (inputFilter != nil) else {
        return
    }
    
    //2nd filter
    let hiddenToOutputWeightsData = BNNSLayerData(
        data:outputWeights, data_type: BNNSDataTypeFloat32,
        data_scale: 0, data_bias: 0, data_table: nil)
    
    let hiddenToOutputBiasData = BNNSLayerData(
        data: outputBiases, data_type: BNNSDataTypeFloat32,
        data_scale: 0, data_bias: 0, data_table: nil)
    
    var hiddenToOutputParams = BNNSFullyConnectedLayerParameters(
        in_size: 2, out_size: 1, weights: hiddenToOutputWeightsData,
        bias: hiddenToOutputBiasData, activation: activation)
    
    var outputDescriptor = BNNSVectorDescriptor(
        size: 1, data_type: BNNSDataTypeFloat32, data_scale: 0, data_bias: 0)
    
    outputFilter = BNNSFilterCreateFullyConnectedLayer(&hiddenDescriptor, &outputDescriptor, &hiddenToOutputParams, nil)
    guard (outputFilter != nil) else {
        return
    }
    
    testNetwork([0, 0])
    testNetwork([0, 1])
    testNetwork([1, 0])
    testNetwork([1, 1])
        
    BNNSFilterDestroy(inputFilter)
    BNNSFilterDestroy(outputFilter)
}

test()