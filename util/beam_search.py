# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 00:47
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : generation_util.py
# @Software: CleanTransformer
# @Description: generation_util

import torch
from .logits_processor import NoRepeatNGramLogitsProcessor, TemperatureLogitsWrapper, TopKLogitsWrapper, TopPLogitsWrapper


class GenerationMixin():
    def __init__(self):
        # 这部分代码不需要执行，因为初始化是在模型代码中声明
        self.gpt = None
        self.version = None
        
        # new
        self.n_layer = 2

    def generate(self, input_ids, attention_mask=None, position_ids=None, segment_ids=None, generation_configs={}, steamers=None):
        beam_size = generation_configs.get('beam_size', 1)
        max_gen_len = generation_configs.get('max_gen_len', 100)
        end_ids = generation_configs.get('end_ids', None)
        pad_id = generation_configs.get('pad_id', 0)
        no_repeat_ngram_size = generation_configs.get('no_repeat_ngram_size', 0)
        self.do_sample = generation_configs.get('do_sample', True)
        temperature = generation_configs.get('temperature', 1.0)
        top_k = generation_configs.get('top_k', 10)
        top_p = generation_configs.get('top_p', 0.8)
        early_stop = generation_configs.get('early_stop', True)

        # end_ids被保存为一个cuda上的张量end_ids_tensor
        if isinstance(end_ids, int): end_ids = [end_ids]
        end_ids_tensor = torch.tensor(list(end_ids)).to(input_ids.device) if end_ids is not None else None

        self.logits_processors = []
        if no_repeat_ngram_size > 1:
            self.logits_processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))

        self.logits_wrapper = []
        self.temperature = temperature
        if self.do_sample and temperature != 1.0:
            self.logits_wrapper.append(TemperatureLogitsWrapper(temperature))
        if self.do_sample and top_k > 0:
            self.logits_wrapper.append(TopKLogitsWrapper(top_k, min_tokens_to_keep=1))
        if self.do_sample and top_p < 1.0:
            self.logits_wrapper.append(TopPLogitsWrapper(top_p, min_tokens_to_keep=1))

        self.steamers = steamers

        if beam_size == 1:
            return self._greedy_search(input_ids, attention_mask, position_ids, segment_ids,
                                       end_ids_tensor, max_gen_len=max_gen_len, pad_id=pad_id)
        else:
            return self._beam_search(input_ids, attention_mask, position_ids, segment_ids,
                                     end_ids_tensor, max_gen_len=max_gen_len, pad_id=pad_id,
                                     beam_size=beam_size, early_stop=early_stop)

    def _greedy_search(self, input_ids, attention_mask, position_ids, segment_ids, end_ids_tensor, max_gen_len, pad_id):
        # step 不能在每次初始化的时候都set为0，否则会死循环
        # 考虑到step的两个作用：1.分离输入；2.结束的条件判断
        # 这里有2中debug的思路：1.统一step 2.分离step
        # 这里使用统一，因此要实现分离的目的，step应该初始化为1
        # 并且条件判断上要配合step，见后文
        step = 1
        print("输入ids的长度： ", input_ids.shape[1])
        bsz = input_ids.size(0)
        # print("bsz: ",bsz) # 64
        max_len = max_gen_len + input_ids.size(-1)
        print("max_len: ", max_len) # 100 + 10
        # 输入长度10，生成后面的100，总长度110
        # k_v_pasts = [None for _ in range(self.config.n_layer)]
        # 未使用config，在init中手动设置self.n_layer = 2
        # k_v_pasts = [None for _ in range(self.n_layer)] 暂时舍弃，测试参数
        # print("k_v_pasts: ", k_v_pasts)
        
        # unfinished_sequences用来标记哪些已经处理了?
        # unfinished_sequences为len=batchsize的tensor，每个元素初始化为1的序列
        # 这里的作用还不明确，其值具体变化与end_ids_tensor有关
        # 若end_ids=None, 则end_ids_tensor也是None
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        
        print("unfinished_sequences: ", unfinished_sequences, "len: ", len(unfinished_sequences)) # 2
        while True:
            # 结束条件判断应该放在递归前面，否则会死循环，并且需要修改，不能用传递的参数step来判断
            # step参数是从0开始递增的
            # 当step达到max_len时，结束
            ############### 结束条件判断 ###############
            print(input_ids.shape[1]) # 序列长度
            # 这里复用的step与上文切分输入的step不一样，但有关，避免混淆，改为temp_len
            # temp_len + step = seq_len，即输入的shape[-1]或shape[1]
            # 
            temp_len = input_ids.shape[1] - 1 # when input len is 10, temp_len = 9
            # 这里or条件使得满足一个即结束
            # end_ids=None时， 未完成标识恒为1，退化为 if temp_len > max_len: break
            # 结束条件为：当当前长度 > 最大长度时，结束
            # temp_len没有对其增加的操作，那么temp_len的值只取决于其初始化的input_ids.shape[1] - 1
            # 那么input_ids的长度增加的操作在哪里实现呢？如果没有，那么就是死循环的递归
            if unfinished_sequences.max() == 0 or temp_len > max_len:
                break
            
            # break for test
            ############### 计算下一个token的hidden_states （会复用k_v_past并更新k_v_past） ###############
            
            hidden_states= self._greedy_search(input_ids[:, step:],
                                                position_ids=None if position_ids is None else position_ids[:, step:],
                                                segment_ids=None if segment_ids is None else segment_ids[:, step:],
                                                attention_mask=attention_mask,
                                                end_ids_tensor=end_ids_tensor,
                                                max_gen_len=max_gen_len,
                                                pad_id=pad_id)
            last_token_hidden_states = hidden_states[0][:, -1, :]

            ############### Logits Penalty ###############
            if len(self.logits_processors) > 0:
                for _processor in self.logits_processors:
                    last_token_hidden_states = _processor(input_ids, last_token_hidden_states)

            if self.do_sample:
                ############### Logits Sampling ###############
                if len(self.logits_wrapper) > 0:
                    for _wrapper in self.logits_wrapper:
                        last_token_hidden_states = _wrapper(input_ids, last_token_hidden_states)
                probs = torch.nn.functional.softmax(last_token_hidden_states, dim=-1)
                step_output = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                ############### 选出得分最高的token ###############
                step_output = torch.argmax(last_token_hidden_states, dim=-1)

            ############### 对于batch中已经结束的case,不管新的token是什么，都换成pad_id ###############
            step_output = step_output * unfinished_sequences + pad_id * (1 - unfinished_sequences)
            ############### 判断batch的每个case是否结束 ###############
            if end_ids_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    step_output.tile(end_ids_tensor.shape[0], 1).ne(end_ids_tensor.unsqueeze(1)).prod(dim=0)
                )

            ############### 得到最新结果（将作为下一次的输入） ###############
            input_ids = torch.concat([input_ids, step_output[:, None]], dim=-1)
            position_ids = None if position_ids is None else torch.concat([position_ids, (position_ids.max(dim=-1).values + 1).view(-1, 1)], dim=-1)
            segment_ids = None if segment_ids is None else torch.concat([segment_ids, segment_ids[:, -1:]], dim=-1)
            # attention_mask = torch.concat([attention_mask, -attention_mask.new_zeros((*attention_mask.shape[:-1], 1))], dim=-1)
            attention_mask = torch.concat([attention_mask, attention_mask[:, -1:]], dim=-1)

            ############### 流式结果输出 ###############
            steamer_finish = False
            if self.steamers is not None:
                self.steamers = self.steamers if isinstance(self.steamers, list) else [self.steamers, ]
                for steamer in self.steamers:
                    if callable(steamer):
                        _finish = steamer(input_ids.view(bsz, 1, -1))
                        steamer_finish = steamer_finish or _finish
            if steamer_finish:
                break

            
           

        return input_ids.view(bsz, 1, -1)

    def _update_beam_infos(self, beam, generated_beam_infos, input_ids, token_indices, next_tokens, probs, end_ids_tensor,
                           pad_token_id,
                           length_penalty=1.0,
                           early_stop=True):
        bsz = next_tokens.shape[0]
        device = input_ids.device
        ############### 保存next_tokens (非end_id)以及来自于哪个beam ###############
        new_indices = torch.zeros((bsz, beam), dtype=token_indices.dtype, device=device)
        new_tokens = torch.zeros((bsz, beam), dtype=next_tokens.dtype, device=device)
        new_probs = torch.zeros((bsz, beam), dtype=probs.dtype, device=device)

        for batch_i in range(bsz):
            candi_generation = generated_beam_infos[batch_i]['candi_generation']
            ############### 如果当前batch_i生成已结束，token替换为pad ###############
            if generated_beam_infos[batch_i]['is_done']:
                new_tokens[batch_i, :] = pad_token_id
                continue

            valid_beam_i = 0
            for beam_i in range(beam):
                if next_tokens[batch_i, beam_i].item() in end_ids_tensor:
                    ############### 对于每个batch_i，首先产生不少于beam_size个候选（每个候选以end_id结尾） ###############
                    if beam_i >= beam: continue  # 在beam_size之后的end_id分数过低，不要
                    choice_idx = beam * batch_i + token_indices[batch_i, beam_i]
                    score = probs[batch_i, beam_i] / (input_ids.shape[-1] ** length_penalty)  # TODO: 这里文本长度是否需要去掉padding？
                    candi_generation.append({"ids": input_ids[choice_idx],
                                             "score": score})
                    ############### 如果候选大于beam_size，则剔除分数最低的候选 ###############
                    if len(candi_generation) > beam:
                        sorted_scores = sorted([(candi['score'], idx) for idx, candi in enumerate(candi_generation)])
                        del candi_generation[sorted_scores[0][1]]
                        generated_beam_infos[batch_i]['worst_score'] = sorted_scores[1][0]
                    else:
                        generated_beam_infos[batch_i]['worst_score'] = min(score, generated_beam_infos[batch_i]['worst_score'])
                else:
                    ############### 没结束前，要尽量保证有beam_size个next_tokens (非end_id)可用于下次输入 ###############
                    new_indices[batch_i, valid_beam_i] = token_indices[batch_i, beam_i]
                    new_tokens[batch_i, valid_beam_i] = next_tokens[batch_i, beam_i]
                    new_probs[batch_i, valid_beam_i] = probs[batch_i, beam_i]
                    valid_beam_i += 1

                if valid_beam_i >= beam:
                    break

            generated_beam_infos[batch_i]['candi_generation'] = candi_generation

            if len(candi_generation) >= beam:
                ############### 结束条件1: 产生beam_size个候选后，且early_stop，则结束 ###############
                if early_stop:
                    generated_beam_infos[batch_i]['is_done'] = True
                    continue
                ############### 结束条件2: 产生beam_size个候选的最低分数，已经比未来可能产生的最大分数更高，则结束 ###############
                next_highest_prob = probs[batch_i].max().item()
                next_highest_score = next_highest_prob / ((input_ids.shape[-1] + 1) ** length_penalty)
                if generated_beam_infos[batch_i]['worst_score'] > next_highest_score:
                    generated_beam_infos[batch_i]['is_done'] = True

        return generated_beam_infos, new_indices, new_tokens, new_probs

    def _beam_topk(self, x_ids, bsz, beam_size, last_token_hidden_states, probs):
        scores = torch.nn.functional.log_softmax(last_token_hidden_states, dim=-1)
        vocab_size = scores.shape[-1]
        probs = probs.view(-1, 1).expand_as(scores)
        if self.do_sample:
            scores = scores + probs * self.temperature
        else:
            scores = scores + probs
        scores = scores.view(bsz, -1)

        if self.do_sample:
            if len(self.logits_wrapper) > 0:
                for _wrapper in self.logits_wrapper:
                    scores = _wrapper(x_ids, scores)
            probs = torch.nn.functional.softmax(scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=2 * beam_size)
            probs = torch.gather(scores, -1, next_tokens)
            probs, _indices = torch.sort(probs, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)
        else:
            probs, next_tokens = scores.topk(2 * beam_size, dim=1, largest=True, sorted=True)

        ############### 确定next_tokens以及来自于哪个beam ###############
        token_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size
        return token_indices, next_tokens, probs

    def _beam_search(self, input_ids, attention_mask, position_ids, segment_ids, end_ids_tensor, max_gen_len, pad_id, beam_size, early_stop):
        bsz = input_ids.size(0)
        max_len = max_gen_len + input_ids.size(-1)
        k_v_pasts = [None for _ in range(self.config.n_layer)]
        step = 0

        ############### 将所有输入扩展成beam_size份 ###############
        input_ids = input_ids.repeat_interleave(beam_size, dim=0)
        position_ids = None if position_ids is None else position_ids.repeat_interleave(beam_size, dim=0)
        attention_mask = attention_mask.repeat_interleave(beam_size, dim=0)
        segment_ids = None if segment_ids is None else segment_ids.repeat_interleave(beam_size, dim=0)

        ############### sentence score初始化 ###############
        probs = torch.zeros((bsz, beam_size), device=input_ids.device)
        probs[:, 1:] = -1e9  # 第一次输入时，每个beam都一样，为防止从每个beam中都选出同一个最大token，第一次只从beam 1中选token

        ############### 记录每个case状态 ###############
        generated_beam_infos = [{'is_done': False, 'worst_score': 1e9, 'candi_generation': []} for _ in range(bsz)]

        while True:
            hidden_states, k_v_pasts = self(input_ids[:, step:],
                                            position_ids=None if position_ids is None else position_ids[:, step:],
                                            segment_ids=None if segment_ids is None else segment_ids[:, step:],
                                            attention_mask=attention_mask,
                                            k_v_pasts=k_v_pasts)
            last_token_hidden_states = hidden_states[0][:, -1, :]

            if len(self.logits_processors) > 0:
                last_token_hidden_states = last_token_hidden_states.view(bsz * beam_size, -1)
                for logit_processor in self.logits_processors:
                    last_token_hidden_states = logit_processor(input_ids, last_token_hidden_states)

            ############### 获取top_k的next_tokens ###############
            token_indices, step_output, probs = self._beam_topk(input_ids, bsz, beam_size, last_token_hidden_states, probs=probs)

            generated_beam_infos, token_indices, step_output, probs = self._update_beam_infos(beam_size, generated_beam_infos, input_ids,
                                                                                              token_indices, step_output, probs,
                                                                                              end_ids_tensor, pad_token_id=pad_id,
                                                                                              early_stop=early_stop)

            def concat_new(value, name):
                if value is None: return None
                value = value.view(bsz, beam_size, -1)
                value = value.gather(1, token_indices[:, :, None].expand_as(value))
                value = value.view(bsz * beam_size, -1)
                if name == 'token':
                    return torch.concat([value, step_output.view(-1)[:, None]], dim=-1)
                elif name == 'position':
                    return torch.concat([value, value[:, -1:] + 1], dim=-1)
                elif name in ('segment', 'attention'):
                    return torch.concat([value, value[:, -1:]], dim=-1)

            ############### 根据next_tokens对应的token_indices, 构造新的输入 ###############
            input_ids = concat_new(input_ids, name='token')
            position_ids = concat_new(position_ids, name='position')
            attention_mask = concat_new(attention_mask, name='attention')
            segment_ids = concat_new(segment_ids, name='segment')

            ############### 选择了不同的beam，k_v_past也要相应变化 ###############
            for i in range(bsz):
                token_indices[i, :] += i * beam_size  # 这一步的原因是token_indices的shape是（bsz,-1），而k_v_past元素的shape是(bsz*beam_size,-1)
            for i, layer_past in enumerate(k_v_pasts):
                _states = []
                for state_past in layer_past:
                    _states.append(state_past.index_select(0, token_indices.view(-1)))
                k_v_pasts[i] = tuple(_states)

            ############### 流式结果输出 ###############
            steamer_finish = False
            if self.steamers is not None:
                self.steamers = self.steamers if isinstance(self.steamers, list) else [self.steamers, ]
                for steamer in self.steamers:
                    if callable(steamer):
                        _finish = steamer(input_ids.view(bsz, beam_size, -1))
                        steamer_finish = steamer_finish or _finish
            if steamer_finish:
                break

            # END判断
            step = input_ids.shape[1] - 1
            if step > max_len:
                break

        return input_ids.view(bsz, beam_size, -1)


if __name__ == "__main__":
    # 设置 PyTorch 的随机种子
    seed = 114514
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    g = GenerationMixin()
    print(g)
    # default beam_size = 1, as greedy search

    batch_size = 2
    seq_len = 10
    input_ids= torch.randint(low=0, high=50, size=(2, 10))

    '''step = 9
    new_input = input_ids[:, step:]
    print("input_ids: ", input_ids)
    print("new_input: ", new_input) '''
    
    print(input_ids.shape[1])

    ret = g.generate(input_ids)