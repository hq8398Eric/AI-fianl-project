import torch
import torch.nn.functional as F
import torch.distributions.multivariate_normal as torchdist
import numpy as np
from tqdm import tqdm
from utils import save
import math


PAD = 0


def Gaussian2D_loss(V_pred, V_trgt):
    """
    Compute NLL on 2D loss. Refer to paper for more details
    """
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, 0] - V_pred[:, 0]
    normy = V_trgt[:, 1] - V_pred[:, 1]

    sx = torch.exp(V_pred[:, 2]) #sx
    sy = torch.exp(V_pred[:, 3]) #sy
    corr = torch.tanh(V_pred[:, 4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.sum(result)
    
    return result


def shotGen_trainer(data_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, config, device="cpu"):
    encode_length = config['encode_length'] - 1         # use the first 3 strokes to the encoder
    record_loss = {
        'total': [],
        'shot': [],
        'area': []
    }

    for epoch in tqdm(range(config['epochs']), desc='Epoch: '):
        encoder.train(), decoder.train()
        total_loss, total_shot_loss, total_area_loss = 0, 0, 0
        total_instance = 0

        for loader_idx, item in enumerate(data_loader):
            batch_input_shot, batch_input_x, batch_input_y, batch_input_player, batch_input_p_area, batch_input_o_area = item[0].to(device), item[1].to(device), item[2].to(device), item[3].to(device), item[4].to(device), item[5].to(device)
            batch_target_shot, batch_target_x, batch_target_y, batch_target_player, batch_target_p_area, batch_target_o_area = item[6].to(device), item[7].to(device), item[8].to(device), item[9].to(device), item[10].to(device), item[11].to(device)
            seq_len, seq_sets = item[12].to(device), item[13].to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_shot = batch_input_shot[:, :encode_length]
            input_x = batch_input_x[:, :encode_length]
            input_y = batch_input_y[:, :encode_length]
            input_player = batch_input_player[:, :encode_length]
            input_p_area = batch_input_p_area[:, :encode_length]
            input_o_area = batch_input_o_area[:, :encode_length]
            encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_p_area, input_o_area)

            input_shot = batch_input_shot[:, encode_length:]
            input_x = batch_input_x[:, encode_length:]
            input_y = batch_input_y[:, encode_length:]
            input_player = batch_input_player[:, encode_length:]
            input_p_area = batch_input_p_area[:, encode_length:]
            input_o_area = batch_input_o_area[:, encode_length:]
            
            target_shot = batch_target_shot[:, encode_length:]
            target_x = batch_target_x[:, encode_length:]
            target_y = batch_target_y[:, encode_length:]
            target_player = batch_target_player[:, encode_length:]
            target_p_area = batch_target_p_area[:, encode_length:]
            target_o_area = batch_target_o_area[:, encode_length:]
            
            output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, input_p_area, input_o_area, encode_local_output, encode_global_A, encode_global_B, target_player)
            
            pad_mask = (input_shot!=PAD)
            output_shot_logits = output_shot_logits[pad_mask]
            target_shot = target_shot[pad_mask]
            output_xy = output_xy[pad_mask]
            target_x = target_x[pad_mask]
            target_y = target_y[pad_mask]

            _, output_shot = torch.topk(output_shot_logits, 1)
            gold_xy = torch.cat((target_x.unsqueeze(-1), target_y.unsqueeze(-1)), dim=-1).to(device, dtype=torch.float)

            total_instance += len(target_shot)

            loss_shot = criterion['entropy'](output_shot_logits, target_shot)
            loss_area = Gaussian2D_loss(output_xy, gold_xy)

            loss = loss_shot + loss_area
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            total_shot_loss += loss_shot.item()
            total_area_loss += loss_area.item()

        total_loss = round(total_loss / total_instance, 4)
        total_shot_loss = round(total_shot_loss / total_instance, 4)
        total_area_loss = round(total_area_loss / total_instance, 4)

        record_loss['total'].append(total_loss)
        record_loss['shot'].append(total_shot_loss)
        record_loss['area'].append(total_area_loss)

    config['total_loss'] = total_loss
    config['total_shot_loss'] = total_shot_loss
    config['total_area_loss'] = total_area_loss
    save(encoder, decoder, config)

    return record_loss

def to_area(pos, device):
    if pos[1] < 0:
        pos[0] *= -1
        pos[1] *= -1
    arr = [1, 7, 2, 5, 8, 6, 3, 9, 4]
    pos[0] += 1.3 # in range [0, 2.6)
    if pos[0] < 0 or pos[0] >= 2.6 or pos[1] < 0 or pos[1] >= 2:
        return torch.tensor(10, device=device)
    ind = math.floor(pos[0] / (2.6 / 3.0)) + math.floor(pos[1] / (2.0 / 3.0)) * 3
    return torch.tensor(arr[ind], device=device)
    
        

def shotgen_generator(given_seq, encoder, decoder, config, samples, device):
    encode_length = config['encode_length'] - 1
    encoder.eval(), decoder.eval()
    generated_shot_logits, generated_area_coordinates = [], []

    with torch.no_grad():
        # encoding stage
        input_shot = given_seq['given_shot'][:encode_length].unsqueeze(0)
        input_x = given_seq['given_x'][:encode_length].unsqueeze(0)
        input_y = given_seq['given_y'][:encode_length].unsqueeze(0)
        input_player = given_seq['given_player'][:encode_length].unsqueeze(0)
        input_p_area = given_seq['given_p_area'][:encode_length].unsqueeze(0)
        input_o_area = given_seq['given_o_area'][:encode_length].unsqueeze(0)
        

        encode_local_output, encode_global_A, encode_global_B = encoder(input_shot, input_x, input_y, input_player, input_p_area, input_o_area)

        for sample_id in range(samples):
            current_generated_shot, current_generated_area = [], []
            total_instance = len(given_seq['given_shot']) - len(given_seq['given_shot'][:encode_length])
            prev_x = torch.tensor(0, device = device)
            prev_y = torch.tensor(0, device = device)
            for seq_idx in range(encode_length, given_seq['rally_length']-1):
                if seq_idx == encode_length:
                    input_shot = given_seq['given_shot'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_x = given_seq['given_x'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_y = given_seq['given_y'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_player = given_seq['given_player'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_p_area = given_seq['given_p_area'][seq_idx].unsqueeze(0).unsqueeze(0)
                    input_o_area = given_seq['given_o_area'][seq_idx].unsqueeze(0).unsqueeze(0)
                    prev_p_area = input_p_area
                    prev_o_area = input_o_area
                else:
                    # use its own predictions as the next input
                    input_shot = torch.cat((input_shot, prev_shot), dim=-1)
                    input_x = torch.cat((input_x, prev_x), dim=-1)
                    input_y = torch.cat((input_y, prev_y), dim=-1)
                    input_player = torch.cat((input_player, prev_player), dim=-1)
                    input_p_area = torch.cat((input_p_area, prev_p_area), dim=-1)
                    input_o_area = torch.cat((input_o_area, prev_o_area), dim=-1)
                    
                target_player = given_seq['target_player'][seq_idx-encode_length].unsqueeze(0).unsqueeze(0)

                output_xy, output_shot_logits = decoder(input_shot, input_x, input_y, input_player, input_p_area, input_o_area, encode_local_output, encode_global_A, encode_global_B, target_player)

                # sample area coordinates
                sx = torch.exp(output_xy[:, -1, 2]) #sx
                sy = torch.exp(output_xy[:, -1, 3]) #sy
                corr = torch.tanh(output_xy[:, -1, 4]) #corr
                
                cov = torch.zeros(2, 2).cuda(output_xy.device)
                cov[0, 0]= sx * sx
                cov[0, 1]= corr * sx * sy
                cov[1, 0]= corr * sx * sy
                cov[1, 1]= sy * sy
                mean = output_xy[:, -1, 0:2]
                
                mvnormal = torchdist.MultivariateNormal(mean, cov)
                output_xy = mvnormal.sample().unsqueeze(0)

                # sampling
                shot_prob = F.softmax(output_shot_logits, dim=-1)
                output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)

                while output_shot[0, -1, 0] == 0:
                    output_shot = shot_prob[0].multinomial(num_samples=1).unsqueeze(0)
                prev_p_area = to_area([prev_x.item(), prev_y.item()], device).unsqueeze(0).unsqueeze(0)
                prev_shot = output_shot[:, -1, :]
                prev_x = output_xy[:, -1, 0].unsqueeze(1)
                prev_y = output_xy[:, -1, 1].unsqueeze(1)
                prev_player = target_player.clone()
                prev_o_area = to_area([prev_x.item(), prev_y.item()], device).unsqueeze(0).unsqueeze(0)
                

                # transform to original format
                ori_shot = config['uniques_type'][prev_shot.item()-1]
                ori_x = prev_x.item()
                ori_y = prev_y.item()

                current_generated_shot.append(shot_prob[0][-1][1:].cpu().tolist())      # 0 is pad
                current_generated_area.append((ori_x, ori_y))

            generated_shot_logits.append(current_generated_shot), generated_area_coordinates.append(current_generated_area)

    return generated_shot_logits, generated_area_coordinates
