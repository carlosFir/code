from transformers import GPT2Model, GPT2Config, GPT2PreTrainedModel, GPT2LMHeadModel # , CausalLMOutputWithCrossAttentions
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .logits_processor import NoRepeatNGramLogitsProcessor, TemperatureLogitsWrapper, TopKLogitsWrapper, TopPLogitsWrapper
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss

def generate_model(conf):
    return ChemGPT(ChemGPTConfig(
        vocab_size=int(conf['data']['vocab_size']),
        n_positions=int(conf['model']['n_positions']),
        n_embd=int(conf['model']['n_embd']),
        n_layer=int(conf['model']['n_layer']),
        n_head=int(conf['model']['n_head']),
        n_props=int(conf['model']['n_props'])
        ))

class ChemGPTConfig(GPT2Config):
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_props = 4,
        n_inner=None,
        alpha=1,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        
        self.n_props = n_props
        self.alpha = alpha
        super().__init__(
            vocab_size = vocab_size,
            n_positions = n_positions,
            n_layer = n_layer,
            n_head = n_head,
            n_inner = n_inner,
            n_embd = n_embd,
            activation_function = activation_function,
            resid_pdrop = resid_pdrop,
            embd_pdrop = embd_pdrop,
            attn_pdrop = attn_pdrop,
            layer_norm_epsilon = layer_norm_epsilon,
            initializer_range = initializer_range,
            summary_type = summary_type,
            summary_use_proj = summary_use_proj,
            summary_activation = summary_activation,
            summary_first_dropout = summary_first_dropout,
            summary_proj_to_labels = summary_proj_to_labels,
            scale_attn_weights = scale_attn_weights,
            use_cache = use_cache,
            scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn = reorder_and_upcast_attn,
            bos_token_id=bos_token_id, 
            eos_token_id=eos_token_id, **kwargs)

class ChemGPT(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.num_props = config.n_props
        self.num_embed = config.n_embd
        self.prop_in_linear = nn.Linear(self.num_props, self.num_embed)  # 性质到embed
        self.prop_out_linear = nn.Linear(self.num_embed, self.num_props) # hidden state到性质
        self.lm_head = nn.Linear(self.num_embed, config.vocab_size, bias=False) # hidden state到token
        self.alpha = config.alpha # 性质预测loss的权重

    def forward(self, data, labels=None, attention_mask=None):
        ''' 
        输入应为性质和完整的字符串数据
        输出应为完整的字符串数据和性质
        data: [props, seqs]    带终止符 b, t-1, d
        labels: [seqs, props]  带终止符 b, t, d
        attention_mask: 标记seqs中哪些是有效输入，哪些是padding B, t-1
        '''
        props, seqs = data[0], data[1]
        prop_embeds = self.prop_in_linear(props) # b, d
        prop_embeds = prop_embeds.unsqueeze(1) # b, 1, d
        batch_size = props.shape[0]
        seq_embeds = self.transformer.wte(seqs)


        inputs_embeds = torch.cat([prop_embeds, seq_embeds], dim=1) # b, t, d
        
        # print(inputs_embeds.shape, attention_mask.shape)
        # exit()
        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        hidden_states = transformer_outputs[0]  # b, t, d
        seq_hidden = hidden_states[:,:,:] 

        if attention_mask is not None: 
            props_index = torch.argmin(attention_mask, dim=-1)-1 # 
        else:
            props_index = -1
        # print(props_index)

        props_hidden = hidden_states[torch.arange(batch_size), props_index,:]

        lm_logits = self.lm_head(seq_hidden)
        props_pred = self.prop_out_linear(props_hidden).squeeze(1) # b, num_props

        # 算loss
        loss = None
        if labels is not None:
            seq_label = labels[0]
            props_label = labels[1]
            # print(lm_logits.shape, seq_label.shape)
            shift_logits = lm_logits[..., 1:, :].contiguous()
            shift_labels = seq_label[..., 1:].contiguous()
            # Flatten the tokens
            loss_seq_fct = CrossEntropyLoss()
            loss_prop_fct = L1Loss()

            loss_seq = loss_seq_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_prop = loss_prop_fct(props_pred, props_label)
            loss = loss_seq + self.alpha * loss_prop

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    
    def _generate_next_token(self, input_ids, past_key_values):
        '''
        生成下一个token的概率和对应的hidden_states
        '''
        '''print(input_ids.shape) # (32, 1)->(b, token)
        print(len(past_key_values))

        print(input_ids)'''
        transformer_outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values)

        hidden_states = transformer_outputs[0]
        seq_hidden = hidden_states[:,:,:]

        lm_logits = self.lm_head(seq_hidden)
        outputs = (lm_logits, hidden_states)
        return outputs, list(transformer_outputs.past_key_values)

    def _generate_past_key_values(self, data): 
        # 输出data中seqs[:,:-1]对应的key和value
        props, seqs = data[0], data[1]
        output = self.forward([props, seqs[..., :-1]])
        past_key_values = output.past_key_values
        return list(past_key_values)
    
    def generate(self, data, attention_mask=None, position_ids=None, segment_ids=None, generation_configs={}, steamers=None):
        beam_size = generation_configs.get('beam_size', 1)
        max_gen_len = generation_configs.get('max_gen_len', 5)
        end_ids = generation_configs.get('end_ids', 1)  # --------------------------要改-----------------------
        pad_id = generation_configs.get('pad_id', 1)  # --------------------------要改-----------------------
        no_repeat_ngram_size = generation_configs.get('no_repeat_ngram_size', 0)
        self.do_sample = generation_configs.get('do_sample', True)
        temperature = generation_configs.get('temperature', 1.0)
        top_k = generation_configs.get('top_k', 10)
        top_p = generation_configs.get('top_p', 0.8)
        early_stop = generation_configs.get('early_stop', True)

        _, input_ids = data[0], data[1]
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
            return self._greedy_search(data, attention_mask, position_ids, segment_ids,
                                       end_ids_tensor, max_gen_len=max_gen_len, pad_id=pad_id)
        else:
            return self._beam_search(data, attention_mask, position_ids, segment_ids,
                                     end_ids_tensor, max_gen_len=max_gen_len, pad_id=pad_id,
                                     beam_size=beam_size, early_stop=early_stop)

    def _greedy_search(self, data, attention_mask, position_ids, segment_ids, end_ids_tensor, max_gen_len, pad_id):
        _, input_ids = data[0], data[1]
        bsz = input_ids.size(0)
        # max_len = max_gen_len + input_ids.size(-1)
        max_len = max_gen_len
        # past_key_values = [None for _ in range(self.config.n_layer)]
        past_key_values = self._generate_past_key_values(data) # 生成除最后一个token外的key-values
        input_ids = input_ids[:, -1:] # 只保留最后一个token
        step = 0
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        while True:
            print(" ############### 计算下一个token的hidden_states##############")
            print("input_ids = ", input_ids.shape)
            print("input_ids[:, step:] = ", input_ids[:, step:].shape)
            print("Target length: {}, temp length: {}".format((max_len+1), input_ids.size(-1)))
            ############### 计算下一个token的hidden_states （会复用k_v_past并更新k_v_past） ###############
            hidden_states, past_key_values = self._generate_next_token(input_ids[:, step:],
                                            past_key_values=past_key_values)
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
            attention_mask = None if attention_mask is None else torch.concat([attention_mask, attention_mask[:, -1:]], dim=-1)

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

            ############### 结束条件判断 ###############
            step = input_ids.shape[1] - 1
            if unfinished_sequences.max() == 0 or step > max_len:
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

    def _beam_search(self, data, attention_mask, position_ids, segment_ids, end_ids_tensor, max_gen_len, pad_id, beam_size, early_stop):
        props, input_ids = data[0], data[1]
        bsz = input_ids.size(0)
        # max_len = max_gen_len + input_ids.size(-1)
        max_len = max_gen_len
        step = 0
           
        ############### 将所有输入扩展成beam_size份 ###############
        input_ids = input_ids.repeat_interleave(beam_size, dim=0)
        props = props.repeat_interleave(beam_size, dim=0)
        # past_key_values = [None for _ in range(self.config.n_layer)]
        past_key_values = self._generate_past_key_values((props, input_ids)) # 生成除最后一个token外的key-values
        input_ids = input_ids[:, -1:] # 只保留最后一个token
        position_ids = None if position_ids is None else position_ids.repeat_interleave(beam_size, dim=0)
        attention_mask = None if attention_mask is None else attention_mask.repeat_interleave(beam_size, dim=0)
        segment_ids = None if segment_ids is None else segment_ids.repeat_interleave(beam_size, dim=0)

        ############### sentence score初始化 ###############
        probs = torch.zeros((bsz, beam_size), device=input_ids.device)
        probs[:, 1:] = -1e9  # 第一次输入时，每个beam都一样，为防止从每个beam中都选出同一个最大token，第一次只从beam 1中选token

        ############### 记录每个case状态 ###############
        generated_beam_infos = [{'is_done': False, 'worst_score': 1e9, 'candi_generation': []} for _ in range(bsz)]

        while True:
            print(" ############### 计算下一个token的hidden_states##############")
            print("input_ids = ", input_ids.shape)
            print("input_ids[:, step:] = ", input_ids[:, step:].shape)
            print("Target length: {}, temp length: {}".format((max_len+1), input_ids.size(-1)))
        
            hidden_states, past_key_values = self._generate_next_token(input_ids[:, step:],
                                            past_key_values=past_key_values)
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
            attention_mask = None if attention_mask is None else concat_new(attention_mask, name='attention')
            segment_ids = concat_new(segment_ids, name='segment')

            ############### 选择了不同的beam，k_v_past也要相应变化 ###############
            for i in range(bsz):
                token_indices[i, :] += i * beam_size  # 这一步的原因是token_indices的shape是（bsz,-1），而k_v_past元素的shape是(bsz*beam_size,-1)
            for i, layer_past in enumerate(past_key_values):
                _states = []
                for state_past in layer_past:
                    _states.append(state_past.index_select(0, token_indices.view(-1)))
                past_key_values[i] = tuple(_states)

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


# 用不了，transformers里的生成模型generate方法输入是ids，我们需要输入是embedding
class ChemGPTLMModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2LMHeadModel(config)
        self.num_props = config.n_props
        self.num_embed = config.n_embd
        self.prop_in_linear = nn.Linear(self.num_props, self.num_embed)  # 性质到embed
        self.prop_out_linear = nn.Linear(self.num_embed, self.num_props) # hidden state到性质
        self.alpha = config.alpha # 性质预测loss的权重

    def forward(self, data, labels=None, attention_mask=None):
        ''' 
        data: [props, seqs]    带终止符 b, t, d
        labels: [seqs, props]  带终止符 b, t, d
        attention_mask: 标记seqs中哪些是有效输入 哪些是padding B, t-1
        todo: attention_mask怎么对齐 返回格式是什么样的
        '''
        props, seqs = data[0], data[1]
        prop_embeds = self.prop_in_linear(props) # b, d
        prop_embeds = prop_embeds.unsqueeze(1) # b, 1, d
        batch_size = props.shape[0]
        seq_embeds = self.transformer.wte(seqs)


        inputs_embeds = torch.cat([prop_embeds, seq_embeds], dim=1) # b, t+1, d
        
        # (loss), lm_logits, presents, (all hidden_states), (attentions)
        transformer_outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # hidden_states = transformer_outputs[0]
        # seq_hidden = hidden_states[:,:,:]

        if attention_mask is not None:
            props_index = torch.argmin(attention_mask, dim=-1)-1 # props对应token的位置索引
        else:
            props_index = -1

        hidden_states = transformer_outputs[3][-1]
        props_hidden = hidden_states[torch.arange(batch_size), props_index,:]

        # lm_logits = self.lm_head(seq_hidden)
        props_pred = self.prop_out_linear(props_hidden).squeeze(1) # b, num_props

        # 算loss
        loss = None
        if labels is not None:
            props_label = labels[1]
            # Flatten the tokens
            loss_prop_fct = L1Loss()

            loss_seq = transformer_outputs[0]
            loss_prop = loss_prop_fct(props_pred, props_label)
            loss = loss_seq + self.alpha * loss_prop

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=transformer_outputs[1],
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    
    def generate(self, data, beam_size=10, max_length=2048):
        pass


if __name__=='__main__':
    config = ChemGPTConfig()

    model = ChemGPT(config).to('cuda:6')
    
    print("here")
    props = torch.randn([1, 4]).to('cuda:6')
    print("here")
    seqs = torch.ones([1, 512]).long().to('cuda:6')


    labels = [seqs, props]
    data = [props, seqs[:,:-1]]

    # attention_mask = torch.zeros(2, 10)
    # attention_mask[0, :5] = 1
    # attention_mask[1, :3] = 1

    # output = model(data, labels, attention_mask)
    # print(output.logits)
    # print(output.attentions)
    print("props: ", props.shape)
    print("seqs: ", seqs.shape)
    print('---------test generation-----------')
    print(model.generate(data))
    