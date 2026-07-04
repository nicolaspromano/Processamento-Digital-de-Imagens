import streamlit as st
import torch
import torch.nn as nn
import numpy as np



#------------------------------------------------------------------------------

#PARA COMPILAR, INSIRA NO TERMINAL: python -m streamlit run app.py

#------------------------------------------------------------------------------



#arquitetura
class CNN3D_OpenBHB(nn.Module):
    def __init__(self, n_classes=1):
        super(CNN3D_OpenBHB, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(256, 64, kernel_size=1, stride=1)

        self.batchnorm1 = nn.BatchNorm3d(32)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.batchnorm3 = nn.BatchNorm3d(128)
        self.batchnorm4 = nn.BatchNorm3d(256)
        self.batchnorm5 = nn.BatchNorm3d(256)
        self.batchnorm6 = nn.BatchNorm3d(64)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool3d(kernel_size=(3, 4, 3), stride=1)
        self.dropout = nn.Dropout3d(p=0.5)
        self.relu = nn.ReLU()
        self.classifier = nn.Conv3d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu(self.maxpool(self.batchnorm1(self.conv1(x))))
        x = self.relu(self.maxpool(self.batchnorm2(self.conv2(x))))
        x = self.relu(self.maxpool(self.batchnorm3(self.conv3(x))))
        x = self.relu(self.maxpool(self.batchnorm4(self.conv4(x))))
        x = self.relu(self.maxpool(self.batchnorm5(self.conv5(x))))

        x = self.relu(self.batchnorm6(self.conv6(x)))
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = x.view(x.shape[0], -1)
        return x

#carrega os pesos
@st.cache_resource
def carregar_modelo():
    modelo = CNN3D_OpenBHB()
    modelo.load_state_dict(torch.load("melhor_modelo_openbhb.pth", map_location=torch.device('cpu')))
    modelo.eval()
    return modelo

modelo = carregar_modelo()

#Streamlit
st.set_page_config(page_title="Laudo - Idade Cerebral", page_icon="🧠", layout="centered")

st.title("🧠 AGE GAP")
st.write("Faça o upload do volume VBM de ressonância magnética e informe a idade cronológica do paciente para calcular o **Brain Age Gap (BAG)**.")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    arquivo_upload = st.file_uploader("Upload do arquivo do paciente (.npy)", type=['npy'])

with col2:
    idade_real = st.number_input("Idade Cronológica (anos)", min_value=0.0, max_value=120.0, value=60.0, step=0.1)

if st.button("Gerar Laudo", type="primary"):
    if arquivo_upload is not None:
        with st.spinner('Processando imagem através da Rede Neural...'):
            try:
                #carrega o arquivo do usuario
                volume_mri = np.load(arquivo_upload)
                tensor_mri = torch.from_numpy(volume_mri).float()
                
                #ajusta para tamanho correto
                tensor_mri = tensor_mri.view(1, 1, 121, 145, 121)

                #inferência
                with torch.no_grad():
                    predicao = modelo(tensor_mri)
                    idade_predita = predicao.item()

                #cálculos
                bag = idade_predita - idade_real
                status = "Acelerado" if bag > 0 else "Preservado"
                cor_status = "red" if bag > 0 else "green"

                #resultados
                st.success("Laudo gerado com sucesso!")
                
                st.subheader("Resultados:")
                res_col1, res_col2, res_col3 = st.columns(3)
                
                res_col1.metric("Idade Cronológica", f"{idade_real:.1f} anos")
                res_col2.metric("Idade Predita", f"{idade_predita:.1f} anos")
                res_col3.metric("Brain Age Gap (BAG)", f"{bag:+.2f} anos")

                st.markdown(f"**Status de Envelhecimento:** :{cor_status}[**{status}**]")

            except Exception as e:
                st.error(f"Erro ao processar o arquivo: {e}")
    else:
        st.warning("Por favor, faça o upload de um arquivo .npy antes de gerar o laudo.")