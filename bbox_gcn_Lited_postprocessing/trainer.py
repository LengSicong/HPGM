from __future__ import print_function
import os
import time
import vutils
import torch
import json
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from six.moves import range
from miscc.config import cfg
from miscc.utils import mkdir_p, bbox_iou, bbox_refiner
from model_graph import GCN, BBOX_NET
from model_graph import get_normalization_2d, get_activation, _get_padding, _init_conv, ResidualBlock, Flatten, Interpolate
plt.switch_backend('agg')


# ################## Shared functions ###################
def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def define_optimizers(model, lr, weight_decay):
    optimizer_model = optim.Adam(model.parameters(),
                            lr=lr, 
                            weight_decay=weight_decay,
                            betas=(0.5, 0.999))
    return optimizer_model


def save_model(model, epoch, model_dir, model_name, best=False):
    if best:
        torch.save(model.state_dict(), '%s/%s_best.pth' % (model_dir, model_name))
    else:
        torch.save(model.state_dict(), '%s/%s_%d.pth' % (model_dir, model_name, epoch))


def save_img_results(imgs_tcpu, real_box, boxes_pred, count, image_dir):
    num = cfg.TRAIN.VIS_COUNT
    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    # real_img = imgs_tcpu[-1][0:num]
    # vutils.save_image(
    #     real_img, '%s/count_%09d_real_samples.png' % (image_dir, count),
    #     normalize=True)
    real_img = None
    # save bounding box images
    vutils.save_bbox(
        real_img, real_box, '%s/count_%09d_real_bbox.png' % (image_dir, count),
        normalize=True)
    vutils.save_bbox(
        real_img, boxes_pred, '%s/count_%09d_fake_bbox.png' % (image_dir, count),
        normalize=True)
    # save floor plan images
    vutils.save_floor_plan(
        real_img, real_box, '%s/count_%09d_real_floor_plan.png' % (image_dir, count),
        normalize=True)
    vutils.save_floor_plan(
        real_img, boxes_pred, '%s/count_%09d_fake_floor_plan.png' % (image_dir, count),
        normalize=True)


def save_img_results_test(imgs_tcpu, real_box, boxes_pred, count, test_dir):
    num = cfg.TRAIN.VIS_COUNT

    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    # real_img = imgs_tcpu[-1][0:num]
    # vutils.save_image(
    #     real_img, '%s/count_%09d_real_samples.png' % (test_dir, count),
    #     normalize=True)
    real_img = None
    # save bounding box images
    vutils.save_bbox(
        real_img, real_box, '%s/count_%09d_real_bbox.png' % (test_dir, count),
        normalize=True)

    vutils.save_bbox(
        real_img, boxes_pred, '%s/count_%09d_fake_bbox.png' % (test_dir, count),
        normalize=True)

    # save floor plan images
    vutils.save_floor_plan(
        real_img, real_box, '%s/count_%09d_real_floor_plan.png' % (test_dir, count),
        normalize=True)

    vutils.save_floor_plan(
        real_img, boxes_pred, '%s/count_%09d_fake_floor_plan.png' % (test_dir, count),
        normalize=True)


def save_img_results_for_FID(imgs_tcpu, fake_imgs, save_path_real, 
                            save_path_fake, step_test, batch_size):
    real_img = imgs_tcpu[-1]
    fake_img = fake_imgs[-1]
    # save image for FID calculation
    vutils.save_image_for_fid(real_img, fake_img, save_path_real, 
                            save_path_fake, step_test, batch_size,
                            normalize=True)


def save_txt_results(text, count, text_dir):
    with open('%s/count_%09d.txt'%(text_dir, count), 'a') as f:
        for t in text:
            f.write('{}\n'.format(t))


def save_txt_results_bbox(boxes, count, text_dir):
    # print(boxes[0][1])
    # assert False
    # room_classes = ['livingroom', 'bedroom', 'corridor', 'kitchen', 
    #                 'washroom', 'study', 'closet', 'storage', 'balcony']
    # rooms_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    room_classes = ['Bathroom', 'DiningRoom', 'StudyRoom', 'MasterRoom', 'SecondRoom', 'Entrance', 'Balcony', 'Kitchen', 'ChildRoom', 'LivingRoom', 'Wall-in', 'Storage', 'GuestRoom']
    rooms_counter = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    temp_dir = {}
    temp_dir["rooms"] = {}
    for i in range(len(boxes[0][1])):
        idx = boxes[0][1][i]
        rooms_counter[idx] += 1
        key = "%s%s" % (room_classes[idx], rooms_counter[idx])
        bounding_box = boxes[0][0][i].cpu().detach().numpy()
        temp_dir["rooms"][key] = {"min_x": "%s"%(bounding_box[0]),
                                "min_y": "%s"%(bounding_box[1]),
                                "max_x": "%s"%(bounding_box[2]),
                                "max_y": "%s"%(bounding_box[3])}
    with open('%s/count_%09d.json'%(text_dir, count), 'w') as f:
        json.dump(temp_dir, f)


# ################# Text to image task############################ #
class LayoutTrainer(object):
    def __init__(self, output_dir, dataloader_train, imsize, dataset_test):
        # build save data dir
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        self.text_dir = os.path.join(output_dir, 'Text')
        self.log_dir = os.path.join(output_dir, 'Log')
        self.image_dir_test = os.path.join(output_dir, 'Image_test')
        self.text_dir_test = os.path.join(output_dir, 'Text_test')
        self.image_dir_eval = os.path.join(output_dir, 'Image_eval')
        self.text_dir_eval = os.path.join(output_dir, 'Text_eval')
        self.text_dir_eval_gt = os.path.join(output_dir, 'Text_eval_gt')
        self.region_dir_eval = os.path.join(output_dir, 'region_process_eval')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        mkdir_p(self.text_dir)
        mkdir_p(self.log_dir)
        mkdir_p(self.image_dir_test)
        mkdir_p(self.text_dir_test)
        mkdir_p(self.image_dir_eval)
        mkdir_p(self.text_dir_eval)
        mkdir_p(self.text_dir_eval_gt)
        mkdir_p(self.region_dir_eval)
        # save the information of cfg
        log_cfg = os.path.join(self.log_dir, 'log_cfg.json')
        with open(log_cfg, 'a') as outfile:
            json.dump(cfg, outfile)
            outfile.write('\n')

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.device = torch.device('cuda:{}'.format(self.gpus[0]) if self.num_gpus>0 else 'cpu')
        cudnn.benchmark = True

        if cfg.TRAIN.FLAG:
            self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        else:
            self.batch_size = cfg.EVAL.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        if dataloader_train!=None:
            self.dataloader_train = dataloader_train
            self.num_batches = len(self.dataloader_train)
        self.dataloader_test = dataset_test
        self.best_loss = 10000.0
        self.best_epoch = 0

    def prepare_data(self, data):
        label_imgs, _, wrong_label_imgs, _, graph, bbox, objs_vector, key, boundary_image = data
        vgraph, vbbox, vobjs_vector, vboundary_image = [], [], [], []
        if cfg.CUDA:
            for i in range(len(graph)):
                # vgraph.append((graph[i][0].to(self.device), graph[i][1].to(self.device)))
                vgraph.append(graph[i].to(self.device))
                vbbox.append((bbox[i][0].to(self.device), bbox[i][1].to(self.device)))
                vobjs_vector.append((objs_vector[i][0].to(self.device), objs_vector[i][1].to(self.device)))
                vboundary_image.append(boundary_image[i].to(self.device))
        return label_imgs, vgraph, vbbox, vobjs_vector, key, vboundary_image

    def prepare_data_test(self, data):
        # imgs, w_imgs, t_embedding, _ = data
        label_imgs, _, graph, bbox, objs_vector, key, boundary_image = data

        # real_vimgs = []
        vgraph, vbbox, vobjs_vector, vboundary_image = [], [], [], []
        if cfg.CUDA:
            for i in range(len(graph)):
                # vgraph.append((graph[i][0].to(self.device), graph[i][1].to(self.device)))
                vgraph.append(graph[i].to(self.device))
                vbbox.append((bbox[i][0].to(self.device), bbox[i][1].to(self.device)))
                vobjs_vector.append((objs_vector[i][0].to(self.device), objs_vector[i][1].to(self.device)))
                vboundary_image.append(boundary_image[i].to(self.device))
        return label_imgs, vgraph, vbbox, vobjs_vector, key, vboundary_image

    def build_cnn(self, arch, normalization='batch', activation='leakyrelu', padding='same',
              pooling='max', init='default'):

        if isinstance(arch, str):
            arch = arch.split(',')
        cur_C = 3
        if len(arch) > 0 and arch[0][0] == 'I':
            cur_C = int(arch[0][1:])
            arch = arch[1:]

        first_conv = True
        flat = False
        layers = []
        for i, s in enumerate(arch):
            if s[0] == 'C':
                if not first_conv:
                    layers.append(get_normalization_2d(cur_C, normalization))
                    layers.append(get_activation(activation))
                first_conv = False
                vals = [int(i) for i in s[1:].split('-')]
                if len(vals) == 2:
                    K, next_C = vals
                    stride = 1
                elif len(vals) == 3:
                    K, next_C, stride = vals
                # K, next_C = (int(i) for i in s[1:].split('-'))
                P = _get_padding(K, padding)
                conv = nn.Conv2d(cur_C, next_C, kernel_size=K, padding=P, stride=stride)
                layers.append(conv)
                _init_conv(layers[-1], init)
                cur_C = next_C
            elif s[0] == 'R':
                norm = 'none' if first_conv else normalization
                res = ResidualBlock(cur_C, normalization=norm, activation=activation,
                                    padding=padding, init=init)
                layers.append(res)
                first_conv = False
            elif s[0] == 'U':
                factor = int(s[1:])
                layers.append(Interpolate(scale_factor=factor, mode='nearest'))
            elif s[0] == 'P':
                factor = int(s[1:])
                if pooling == 'max':
                    pool = nn.MaxPool2d(kernel_size=factor, stride=factor)
                elif pooling == 'avg':
                    pool = nn.AvgPool2d(kernel_size=factor, stride=factor)
                layers.append(pool)
            elif s[:2] == 'FC':
                _, Din, Dout = s.split('-')
                Din, Dout = int(Din), int(Dout)
                if not flat:
                    layers.append(Flatten())
                flat = True
                layers.append(nn.Linear(Din, Dout))
                if i + 1 < len(arch):
                    layers.append(get_activation(activation))
                cur_C = Dout
            else:
                raise ValueError('Invalid layer "%s"' % s)
        layers = [layer for layer in layers if layer is not None]
        # for layer in layers:
        #   print(layer)
        return nn.Sequential(*layers), cur_C

    def define_models(self):
        if cfg.TRAIN.USE_SIZE_AS_INPUT:
            # objs_vector_dim = 19
            objs_vector_dim = 23
        else:
            objs_vector_dim = 18
        # build gcn model
        input_graph_dim = objs_vector_dim
        hidden_graph_dim = 64
        output_graph_dim = objs_vector_dim
        gcn = GCN(nfeat=input_graph_dim, 
                  nhid=hidden_graph_dim, output_dim=output_graph_dim)
        # build box_net model
        gconv_dim = objs_vector_dim
        gconv_hidden_dim = 512
        box_net_dim = 4
        mlp_normalization = 'none'
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        box_net = BBOX_NET(box_net_layers, batch_norm=mlp_normalization)
        ''' inside_cnn '''
        input_dim=3
        inside_cnn_arch="C3-32-2,C3-64-2,C3-128-2,C3-256-2"
        inside_cnn,inside_feat_dim = self.build_cnn(
            f'I{input_dim},{inside_cnn_arch}',
            padding='valid'
        )
        inside_cnn = nn.Sequential(
        inside_cnn,
        nn.AdaptiveAvgPool2d(1)
        )
        boundary_mlp = nn.Sequential(nn.Linear(inside_feat_dim, 23), nn.ReLU())
        # nn.Linear(inside_feat_dim, 23)
        return gcn, box_net, inside_cnn, boundary_mlp


    def train(self):
        # plot
        self.training_epoch = []
        self.testing_epoch = []
        self.training_error = []
        self.testing_error = []
        # define models
        if cfg.TRAIN.USE_GCN:
            self.gcn, self.box_net, self.inside_cnn, self.boundary_mlp = self.define_models()
        else:
            _, self.box_net, self.inside_cnn, self.boundary_mlp = self.define_models()

        # load gcn checkpoints
        if cfg.TRAIN.GCN != '':
            self.gcn.load_state_dict(
                torch.load(cfg.TRAIN.GCN))
        # load box_net checkpoints
        if cfg.TRAIN.BOX_NET != '':
            self.box_net.load_state_dict(
                torch.load(cfg.TRAIN.BOX_NET))

        # optimization method
        self.optimizer_gcn = define_optimizers(
            self.gcn, cfg.GCN.LR, cfg.GCN.WEIGHT_DECAY)
        self.optimizer_bbox = define_optimizers(
            self.box_net, cfg.BBOX.LR, cfg.BBOX.WEIGHT_DECAY)
        self.optimizer_inside_cnn = define_optimizers(
            self.inside_cnn, cfg.BBOX.LR, cfg.BBOX.WEIGHT_DECAY)
        self.optimizer_boundary_mlp = define_optimizers(
            self.boundary_mlp, cfg.BBOX.LR, cfg.BBOX.WEIGHT_DECAY)

        # criterion function
        self.criterion_bbox = nn.MSELoss()
        if cfg.CUDA:
            # criterion
            self.criterion_bbox.to(self.device)
            # model
            self.gcn.to(self.device)
            self.box_net.to(self.device)
            self.inside_cnn.to(self.device)
            self.boundary_mlp.to(self.device)
        predictions = []
        start_epoch = 0
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            # ================== #
            #      Training      #
            # ================== #
            for step, data in enumerate(self.dataloader_train, 0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.imgs_tcpu, self.graph, self.real_box, self.objs_vector, self.key, self.boundary_image = self.prepare_data(data)
                #######################################################
                # (1) Generate layout position
                ######################################################
                self.gcn.train() if cfg.TRAIN.USE_GCN else None
                self.box_net.train()
                self.inside_cnn.train()
                self.boundary_mlp.train()
                ##TODO: Sicong for boundary encoding
                inside_boundary_vectors = self.inside_cnn(torch.cat([tensor.unsqueeze(0) for tensor in self.boundary_image], dim=0)).view(len(self.real_box), -1)
                # for each image
                for i in range(len(self.real_box)):
                    graph_objs_vector = self.gcn(self.objs_vector[i][0], self.graph[i])
                    inside_boundary_vector = self.boundary_mlp(inside_boundary_vectors[i])
                    inside_boundary_vector = torch.cat([inside_boundary_vector.unsqueeze(0)]*graph_objs_vector.shape[0], dim=0)
                    graph_objs_vector = torch.add(graph_objs_vector, inside_boundary_vector*cfg.TRAIN.COEFF.BOUNDARY_VEC)
                    boxes_pred = self.box_net(self.objs_vector[i][0], graph_objs_vector)
                    # optimization
                    if i == 0:
                        err_bbox = self.criterion_bbox(boxes_pred, self.real_box[i][0])
                    else:
                        err_bbox += self.criterion_bbox(boxes_pred, self.real_box[i][0])
                err_bbox = err_bbox / len(self.real_box)
                err_total = cfg.TRAIN.COEFF.BBOX_LOSS * err_bbox
                self.optimizer_gcn.zero_grad()
                self.optimizer_bbox.zero_grad()
                self.optimizer_inside_cnn.zero_grad()
                self.optimizer_boundary_mlp.zero_grad()
                err_total.backward()
                self.optimizer_gcn.step()
                self.optimizer_bbox.step()
                self.optimizer_inside_cnn.step()
                self.optimizer_boundary_mlp.step()

            # save the best models
            print('comparing total loss...')
            if err_total.item() < self.best_loss:
                self.best_loss = err_total.item()
                self.best_epoch = epoch
                print('saving best models...')
                save_model(model=self.gcn, epoch=epoch, model_dir=self.model_dir,
                           model_name='gcn', best=True)
                save_model(model=self.box_net, epoch=epoch, model_dir=self.model_dir,
                           model_name='box_net', best=True)
                save_model(model=self.inside_cnn, epoch=epoch, model_dir=self.model_dir,
                            model_name='inside_cnn', best=True)
                save_model(model=self.boundary_mlp, epoch=epoch, model_dir=self.model_dir,
                            model_name='boundary_mlp', best=True)
            print('\033[1;31m current_epoch[{}] current_loss[{}] \033[0m \033[1;34m best_epoch[{}] best_loss[{}] \033[0m'.format(
                    epoch, err_total.item(), self.best_epoch, self.best_loss))

            # ================= #
            #      Valid        #
            # ================= #
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                self.gcn.eval()
                self.box_net.eval()
                self.inside_cnn.eval()
                self.boundary_mlp.eval()
                boxes_pred_collection = []
                for i in range(len(self.real_box)):
                    graph_objs_vector = self.gcn(self.objs_vector[i][0], self.graph[i])
                    # bounding box prediction
                    boxes_pred_save = self.box_net(self.objs_vector[i][0], graph_objs_vector)
                    boxes_pred_collection.append((boxes_pred_save, self.real_box[i][1]))
                save_img_results(self.imgs_tcpu, self.real_box, boxes_pred_collection, epoch, self.image_dir)
                save_txt_results(self.key, epoch, self.text_dir)

                # evaluate the model
                print('generating the test data...')
                for step_test, data_test in enumerate(self.dataloader_test, 0):
                    # get data
                    self.imgs_tcpu_test, self.graph_test, self.bbox_test, self.objs_vector_test, self.key_test, self.boundary_image = self.prepare_data_test(data_test)
                    boxes_pred_test_collection = []
                    inside_boundary_vectors_test = self.inside_cnn(torch.cat([tensor.unsqueeze(0) for tensor in self.boundary_image], dim=0)).view(len(self.real_box), -1)
                    for i in range(len(self.bbox_test)):
                        graph_objs_vector_test = self.gcn(self.objs_vector_test[i][0], self.graph_test[i])
                        inside_boundary_vector_test = self.boundary_mlp(inside_boundary_vectors_test[i])
                        inside_boundary_vector_test = torch.cat([inside_boundary_vector_test.unsqueeze(0)]*graph_objs_vector_test.shape[0], dim=0)
                        graph_objs_vector_test = torch.add(graph_objs_vector_test, inside_boundary_vector_test*cfg.TRAIN.COEFF.BOUNDARY_VEC)
                        # bounding box prediction
                        boxes_pred_test = self.box_net(self.objs_vector_test[i][0], graph_objs_vector_test)
                        boxes_pred_test_collection.append((boxes_pred_test, self.bbox_test[i][1]))
                        # record the loss
                        if i == 0:
                            err_bbox_test = self.criterion_bbox(boxes_pred_test, self.bbox_test[i][0])
                        else:
                            err_bbox_test += self.criterion_bbox(boxes_pred_test, self.bbox_test[i][0])
                    err_bbox_test = err_bbox_test / len(self.bbox_test)
                    err_total_test = cfg.TRAIN.COEFF.BBOX_LOSS * err_bbox_test
                    if step_test == 0:
                        save_img_results_test(self.imgs_tcpu_test, self.bbox_test, boxes_pred_test_collection, epoch, self.image_dir_test)
                        save_txt_results(self.key_test, epoch, self.text_dir_test)
                    break
                # plot
                self.testing_epoch.append(epoch)
                self.testing_error.append(err_total_test)
            # ================ #
            #      Saving      #
            # ================ #
            self.training_epoch.append(epoch)
            self.training_error.append(err_total)
            # plot
            plt.figure(0)
            plt.plot(self.training_epoch, [train_error.cpu().detach().numpy() for train_error in self.training_error], color="r", linestyle="-", linewidth=1, label="training")
            plt.plot(self.testing_epoch, [test_error.cpu().detach().numpy() for test_error in self.testing_error], color="b", linestyle="-", linewidth=1, label="testing")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc='best')
            plt.savefig(os.path.join(self.output_dir, "loss.png"))
            plt.close(0)

            # loss
            with open(os.path.join(self.log_dir, 'log_loss.txt'), 'a') as f:
                f.write('{},{}\n'.format(epoch, self.training_error[-1]))
            # print
            end_t = time.time()
            try:
                print('[%d/%d][%d] Loss_total: %.5f Loss_bbox: %.5f Time: %.2fs' % (epoch, self.max_epoch,
                      self.num_batches, err_total, cfg.TRAIN.COEFF.BBOX_LOSS*err_bbox, end_t - start_t))
            except IOError as e:
                print(e)
                pass
            # for cfg.TRAIN.CHECK_POINT_INTERVAL times save one model
            print('saving checkpoint models...')
            if epoch % cfg.TRAIN.CHECK_POINT_INTERVAL == 0:
                save_model(model=self.gcn, epoch=epoch, model_dir=self.model_dir,
                           model_name='gcn', best=False)
                save_model(model=self.box_net, epoch=epoch, model_dir=self.model_dir,
                           model_name='box_net', best=False)
                save_model(model=self.inside_cnn, epoch=epoch, model_dir=self.model_dir,
                           model_name='inside_cnn', best=False)
                save_model(model=self.boundary_mlp, epoch=epoch, model_dir=self.model_dir,
                           model_name='boundary_mlp', best=False)

    # evaluate the trained models
    def evaluate(self):
        # define models
        self.gcn, self.box_net = self.define_models()
        # load gcn
        if cfg.EVAL.GCN != '':
            self.gcn.load_state_dict(
                torch.load(os.path.join(cfg.EVAL.OUTPUT_DIR, 'Model',
                           cfg.EVAL.GCN)))
        # load box_net
        if cfg.EVAL.BOX_NET != '':
            self.box_net.load_state_dict(
                torch.load(os.path.join(cfg.EVAL.OUTPUT_DIR, 'Model',
                           cfg.EVAL.BOX_NET)))
        # load inside_cnn
        self.inside_cnn.load_state_dict(
            torch.load(os.path.join(cfg.EVAL.OUTPUT_DIR, 'Model',
                          cfg.EVAL.INSIDE_CNN)))
        # load boundary_mlp
        self.boundary_mlp.load_state_dict(
            torch.load(os.path.join(cfg.EVAL.OUTPUT_DIR, 'Model',
                            cfg.EVAL.BOUNDARY_MLP)))
        if cfg.CUDA:
            self.gcn.to(self.device)
            self.box_net.to(self.device)
            self.inside_cnn.to(self.device)
            self.boundary_mlp.to(self.device)
        self.gcn.eval()
        self.box_net.eval()
        self.inside_cnn.eval()
        self.boundary_mlp.eval()

        # evaluate the model
        print('evaluating the test data...')
        total_IoU = 0.0
        count_boxes = 0
        for step_test, data_test in enumerate(self.dataloader_test, 1):
            # get data
            self.imgs_tcpu_test, self.graph_test, self.bbox_test, self.objs_vector_test, self.key_test, self.boundary_image = self.prepare_data_test(data_test)
            boxes_pred_test_collection = []
            boxes_pred_test_collection_gt = []
            inside_boundary_vectors_test = self.inside_cnn(torch.cat([tensor.unsqueeze(0) for tensor in self.boundary_image], dim=0)).view(len(self.real_box), -1)
            for i in range(len(self.bbox_test)):
                graph_objs_vector_test = self.gcn(self.objs_vector_test[i][0], self.graph_test[i])
                inside_boundary_vector_test = self.boundary_mlp(inside_boundary_vectors_test[i])
                inside_boundary_vector_test = torch.cat([inside_boundary_vector_test.unsqueeze(0)]*graph_objs_vector_test.shape[0], dim=0)
                graph_objs_vector_test = torch.add(graph_objs_vector_test, inside_boundary_vector_test*cfg.TRAIN.COEFF.BOUNDARY_VEC)
                # bounding box prediction
                boxes_pred_test = self.box_net(self.objs_vector_test[i][0], graph_objs_vector_test)
                IoU, num_boxes = bbox_iou(boxes_pred_test, self.bbox_test[i][0])
                total_IoU += IoU
                count_boxes += num_boxes
                boxes_pred_test_collection.append((boxes_pred_test, self.bbox_test[i][1]))
                boxes_pred_test_collection_gt.append((self.bbox_test[i][0], self.bbox_test[i][1]))
            # save layout images and texts
            save_img_results_test(self.imgs_tcpu_test, self.bbox_test, boxes_pred_test_collection, step_test, self.image_dir_eval)
            save_txt_results_bbox(boxes_pred_test_collection, step_test, self.text_dir_eval)
            save_txt_results_bbox(boxes_pred_test_collection_gt, step_test, self.text_dir_eval_gt)
            ## region Processing
            from regionProcessing import RegionProcessor,get_merge_image
            # room_classes = ['livingroom', 'bedroom', 'corridor', 'kitchen',
            #                 'washroom', 'study', 'closet', 'storage', 'balcony']
            room_classes = ['Bathroom', 'DiningRoom', 'StudyRoom', 'MasterRoom', 'SecondRoom', 'Entrance', 'Balcony', 'Kitchen', 'ChildRoom', 'LivingRoom', 'Wall-in', 'Storage', 'GuestRoom']
            coord_data = [boxes_pred_test.cpu(), self.bbox_test[0][1], room_classes]
            processor = RegionProcessor(coord_data=coord_data)
            lines, rooms = processor.get_lines_from_json()
            # print(lines, rooms)
            ## post-processing to make the floor plan reasonalbe
            # get_merge_image(lines, rooms, processor, self.region_dir_eval, step_test)
        print('Avg IoU: {}'.format(total_IoU/count_boxes))
