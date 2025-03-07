import yaml
import argparse

config_file_path = "configs/finetune/gconv_adapter.yaml"

# Function to update the YAML configuration
def update_config(args):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the configuration with arguments from argparse
    config['adapter']['bottleneck_size'] = args.bottleneck_size
    config['adapter']['conv_type'] = args.conv_type
    config['adapter']['non_linearity'] = args.non_linearity
    config['adapter']['normalization'] = args.normalization
    config['adapter']['learnable_scalar'] = True if args.learnable_scalar == "True" else False
    config['adapter']['skip_connection'] = True if args.skip_connection == "True" else False
    config['adapter']['normalize'] = True if args.normalize == "True" else False  
    config['positions'] = args.positions
    config['type'] = args.type

    # Write the updated configuration back to the file
    with open(config_file_path, 'w') as file:
        yaml.dump(config, file)

# Main function
def main():
    # Argument parser
    parser = argparse.ArgumentParser()

    # Add arguments corresponding to GConv-Adapter parameters
    parser.add_argument('--bottleneck_size', type=int, default=16, help='Size of the bottleneck layer.')
    parser.add_argument('--conv_type', type=str, default='gcn', choices=['gcn', 'gat'], help='Type of graph convolution.')
    parser.add_argument('--non_linearity', type=str, default='relu', choices=['relu', 'silu', 'none'], help='Type of non-linearity.')
    parser.add_argument('--normalization', type=str, default='batch_norm', choices=['batch_norm', 'layer_norm', 'none'], help='Type of normalization to use.')
    parser.add_argument('--learnable_scalar', type=str, default='True', help='Whether to use a learnable scalar to multiply the output.')
    parser.add_argument('--skip_connection', type=str, default='True', help='Whether to include a skip connection in the forward pass.')
    parser.add_argument('--normalize', type=str, default='True', help='Whether to add self-loops and compute symmetric normalization coefficients.')  
    parser.add_argument('--positions', nargs='+', default=['pre', 'post'], help='List of positions (pre, post).')
    parser.add_argument('--type', type=str, default='sequential', help='Type of adapter (sequential, parallel).')

    args = parser.parse_args()

    # Update the config file with the provided arguments
    update_config(args)

if __name__ == "__main__":
    main()
