from src.models.transductive.gnns import *
from src.models.transductive.nodeformer import *
from src.models.transductive.difformer import *
from src.dataset.transductive.data_utils import normalize

def parse_method(cfg, dataset, num_nodes, num_classes, num_feats, device):
    if cfg.model.method == 'link':
        model = LINK(num_nodes, num_classes).to(device)
    elif cfg.model.method == 'gcn':
        if cfg.dataset.name == 'ogbn-proteins':
            # Pre-compute GCN normalization.
            dataset.graph['edge_index'] = normalize(dataset.graph['edge_index'])
            model = GCN(in_channels=num_feats,
                        hidden_channels=cfg.model.hidden_channels,
                        out_channels=num_classes,
                        dropout=cfg.model.dropout,
                        save_mem=True,
                        use_bn=cfg.model.use_bn).to(device)
        else:
            model = GCN(in_channels=num_feats,
                        hidden_channels=cfg.model.hidden_channels,
                        out_channels=num_classes,
                        num_layers=cfg.model.num_layers,
                        dropout=cfg.model.dropout,
                        use_bn=cfg.model.use_bn).to(device)
    elif cfg.model.method in ['mlp', 'cs']:
        model = MLP(in_channels=num_feats, 
                    hidden_channels=cfg.model.hidden_channels,
                    out_channels=num_classes, 
                    num_layers=cfg.model.num_layers,
                    dropout=cfg.model.dropout).to(device)
    elif cfg.model.method == 'sgc':
        if cfg.gnn_baseline.cached:
            model = SGC(in_channels=num_feats, out_channels=num_classes, hops=cfg.gnn_baseline.hops).to(device)
        else:
            model = SGCMem(in_channels=num_feats, out_channels=num_classes, hops=cfg.gnn_baseline.hops).to(device)
    elif cfg.model.method == 'gprgnn':
        model = GPRGNN(num_feats, cfg.model.hidden_channels, num_classes, alpha=cfg.gnn_baseline.gpr_alpha).to(device)
    elif cfg.model.method == 'appnp':
        model = APPNP_Net(num_feats, cfg.model.hidden_channels, num_classes, alpha=cfg.gnn_baseline.gpr_alpha).to(device)
    elif cfg.model.method == 'gat':
        model = GAT(num_feats, cfg.model.hidden_channels, num_classes, 
                    num_layers=cfg.model.num_layers, 
                    dropout=cfg.model.dropout, 
                    use_bn=cfg.model.use_bn, 
                    heads=cfg.gnn_baseline.gat_heads, 
                    out_heads=cfg.gnn_baseline.out_heads).to(device)
    elif cfg.model.method == 'lp':
        mult_bin = cfg.dataset.name == 'ogbn-proteins'
        model = MultiLP(num_classes, cfg.gnn_baseline.lp_alpha, cfg.gnn_baseline.hops, mult_bin=mult_bin)
    elif cfg.model.method == 'mixhop':
        model = MixHop(num_feats, cfg.model.hidden_channels, num_classes, 
                       num_layers=cfg.model.num_layers,
                       dropout=cfg.model.dropout, hops=cfg.gnn_baseline.hops).to(device)
    elif cfg.model.method == 'gcnjk':
        model = GCNJK(num_feats, cfg.model.hidden_channels, num_classes, 
                      num_layers=cfg.model.num_layers,
                      dropout=cfg.model.dropout, 
                      jk_type=cfg.gnn_baseline.jk_type).to(device)
    elif cfg.model.method == 'gatjk':
        model = GATJK(num_feats, cfg.model.hidden_channels, num_classes, 
                      num_layers=cfg.model.num_layers, 
                      dropout=cfg.model.dropout, 
                      heads=cfg.gnn_baseline.gat_heads,
                      jk_type=cfg.gnn_baseline.jk_type).to(device)
    elif cfg.model.method == 'h2gcn':
        model = H2GCN(num_feats, cfg.model.hidden_channels, num_classes, 
                      dataset.graph['edge_index'], dataset.graph['num_nodes'], 
                      num_layers=cfg.model.num_layers, 
                      dropout=cfg.model.dropout, 
                      num_mlp_layers=cfg.gnn_baseline.num_mlp_layers).to(device)
    elif cfg.model.method == 'nodeformer':
        model = NodeFormer(num_feats, cfg.model.hidden_channels, num_classes, 
                           num_layers=cfg.model.num_layers, 
                           dropout=cfg.model.dropout, 
                           num_heads=cfg.model.num_heads, 
                           use_bn=cfg.model.use_bn, 
                           nb_random_features=cfg.model.M, 
                           use_gumbel=cfg.model.use_gumbel, 
                           use_residual=cfg.model.use_residual, 
                           use_act=cfg.model.use_act, 
                           use_jk=cfg.model.use_jk, 
                           nb_gumbel_sample=cfg.model.K, 
                           rb_order=cfg.model.rb_order, 
                           rb_trans=cfg.model.rb_trans).to(device)
    elif cfg.model.method == 'difformer':
        model = DIFFormer(num_feats, cfg.model.hidden_channels, num_classes, 
                          num_layers=cfg.model.num_layers, 
                          num_heads=cfg.model.num_heads, 
                          kernel=cfg.model.kernel,
                          alpha=cfg.model.alpha, 
                          dropout=cfg.model.dropout, 
                          use_bn=cfg.model.use_bn, 
                          use_residual=cfg.model.use_residual,
                          use_weight=cfg.model.use_weight,
                          use_graph=cfg.model.use_graph,
        )
    else:
        raise ValueError('Invalid method')
    
    return model