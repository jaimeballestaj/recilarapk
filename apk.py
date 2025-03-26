import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Reciclaje Inteligente",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-family: 'Arial Black', sans-serif;
        color: #ffffff;
        text-align: center;
    }
    .subheader {
        font-family: 'Arial', sans-serif;
        font-size: 1.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #1e6b52;
        border-left: 5px solid #1e6b52;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1rem 0;
        color: white;
    }
    .caneca-info {
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #ffffff;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #555;
    }
    .stat-box {
        background-color: #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        color: white;
    }
    /* Mejorar visibilidad general */
    h1, h2, h3, h4, h5, h6, p, li, label, .stButton>button {
        color: white !important;
    }
    /* Estilo para botones más visibles */
    .stButton>button {
        background-color: #1e6b52 !important;
        color: white !important;
        border: 2px solid #25a67a !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
        transition: all 0.3s !important;
    }
    .stButton>button:hover {
        background-color: #25a67a !important;
        border-color: #1e6b52 !important;
        transform: translateY(-2px) !important;
    }
</style>
""", unsafe_allow_html=True)
residuos = pd.DataFrame({
    'residuo': [
        # Plásticos
        'botella plástica', 'vaso de plástico', 'bolsa plástica', 'envase de yogurt', 'cubiertos de plástico',
        'cepillo de dientes', 'juguete de plástico', 'tubo de PVC', 'botella de detergente',
        # Orgánicos
        'cáscara de banana', 'cáscara de naranja', 'restos de comida', 'restos de verduras', 'café molido usado',
        'cáscara de huevo', 'servilleta usada con comida', 'restos de frutas', 'restos de pan',
        'ramitas y hojas', 'pelo/cabello',
         # Papel y cartón
        'papel', 'cartón', 'revista', 'periódico', 'caja de cartón', 'cuaderno usado', 'libro viejo',
        'folleto publicitario', 'rollo de papel higiénico', 'caja de cereal',
         # Vidrio
        'botella de vidrio', 'frasco de vidrio', 'vaso de vidrio roto', 'espejo roto', 'bombilla tradicional',
         # Metal
        'lata de refresco', 'lata de conservas', 'papel aluminio', 'aerosol vacío', 'clips metálicos',
        'grapas', 'clavo oxidado', 'tapas metálicas',
         # No reciclables comunes
        'vaso de icopor', 'envoltura metalizada', 'pañal usado', 'toalla higiénica', 'hilo dental usado',
        'envoltorio de dulces', 'chicle usado', 'colilla de cigarrillo', 'esponja de cocina',
         # Residuos especiales/peligrosos
        'pila/batería', 'medicamento vencido', 'jeringa usada', 'bombilla ahorradora', 'termómetro roto',
        'pintura', 'aceite de motor', 'pesticida', 'tinta de impresora'
    ],
    'tipo': [
        # Plásticos
        'reciclable', 'reciclable', 'reciclable', 'reciclable', 'reciclable',
        'no reciclable', 'no reciclable', 'no reciclable', 'reciclable',
         # Orgánicos
        'organico', 'organico', 'organico', 'organico', 'organico',
        'organico', 'organico', 'organico', 'organico',
        'organico', 'organico',
         # Papel y cartón
        'reciclable', 'reciclable', 'reciclable', 'reciclable', 'reciclable', 'reciclable', 'reciclable',
        'reciclable', 'reciclable', 'reciclable',
         # Vidrio
        'reciclable', 'reciclable', 'no reciclable', 'no reciclable', 'no reciclable',
         # Metal
        'reciclable', 'reciclable', 'reciclable', 'peligroso', 'reciclable',
        'reciclable', 'reciclable', 'reciclable',
         # No reciclables comunes
        'no reciclable', 'no reciclable', 'no reciclable', 'no reciclable', 'no reciclable',
        'no reciclable', 'no reciclable', 'no reciclable', 'no reciclable',
         # Residuos especiales/peligrosos
        'peligroso', 'peligroso', 'peligroso', 'peligroso', 'peligroso',
        'peligroso', 'peligroso', 'peligroso', 'peligroso'
    ]
})
# Mapeo de tipos de residuos a colores de caneca
mapa_caneca = {
    'reciclable': 'azul',
    'organico': 'verde',
    'no reciclable': 'gris',
    'peligroso': 'roja'
}
residuos['caneca'] = residuos['tipo'].map(mapa_caneca)
# Características para el modelo
residuos['es_plastico'] = residuos['residuo'].str.contains(
    "plást|plast|icop|bolsa|pet|pvc|poliestireno|envase|detergente|botella|vaso|juguete", 
    case=False, regex=True
).astype(int)
residuos['es_organico'] = residuos['residuo'].str.contains(
    "banana|naranja|comida|casc|fruta|vegetal|pan|café|huevo|servilleta|pelo|cabello|hoja|ramita",
    case=False, regex=True
).astype(int)
residuos['es_papel'] = residuos['residuo'].str.contains(
    "papel|revista|carton|cartón|periódico|cuaderno|libro|folleto|higiénico|cereal",
    case=False, regex=True
).astype(int)
residuos['es_vidrio'] = residuos['residuo'].str.contains(
    "vidrio|cristal|espejo|bombilla",
    case=False, regex=True
).astype(int)

residuos['es_metal'] = residuos['residuo'].str.contains(
    "lata|metal|aluminio|aerosol|clip|grapa|clavo|tapa",
    case=False, regex=True
).astype(int)
residuos['es_peligroso'] = residuos['residuo'].str.contains(
    "pila|batería|medicamento|jeringa|bombilla ahorradora|termómetro|pintura|aceite|pesticida|tinta|peligroso|tóxico|químico",
    case=False, regex=True
).astype(int)
residuos['es_higienico'] = residuos['residuo'].str.contains(
    "pañal|toalla higiénica|hilo dental|cepillo|esponja",
    case=False, regex=True
).astype(int)

# Preparación del modelo
X = residuos[['es_plastico', 'es_organico', 'es_papel', 'es_vidrio', 'es_metal', 'es_peligroso', 'es_higienico']]
y = residuos['caneca']

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/992/992700.png", width=100)
    st.title("Navegación")
    
    if 'pagina' not in st.session_state:
        st.session_state['pagina'] = "Inicio"
    
    pagina = st.radio(
        "Selecciona una sección:",
        ["Inicio", "Clasificador", "Aprende más", "Estadísticas"],
        key="nav_radio",
        index=["Inicio", "Clasificador", "Aprende más", "Estadísticas"].index(st.session_state['pagina'])
    )
    
    # Actualizar la página cuando cambia la selección
    if pagina != st.session_state['pagina']:
        st.session_state['pagina'] = pagina
        st.rerun()
    
   
    st.markdown("### Contacto")
    st.markdown("Jaime Ballesta J")
    st.markdown("📧 jaimeballestaj@gmail.com")
    st.markdown("📱 +57 3012908001")
    

# Configuración de colores y mensajes para las canecas
color_map = {
    "azul": ("♻️ El residuo debe ir en la caneca AZUL (reciclaje)", "#0066CC", 
             "Estos residuos se pueden transformar en nuevos productos."),
    
    "verde": ("🌿 El residuo debe ir en la caneca VERDE (orgánico)", "#33A532", 
              "Estos residuos se pueden compostar para generar abono natural."),
    
    "gris": ("🗑️ El residuo debe ir en la caneca GRIS (no reciclable)", "#555555", 
             "Estos residuos no se pueden reciclar ni compostar."),
    
    "roja": ("⚠️ ¡Este residuo es PELIGROSO y debe ir en la caneca ROJA!", "#B30000", 
             "Estos residuos requieren manejo especial. Llévalos a un punto de recolección autorizado.")
}

# Página de inicio
if pagina == "Inicio":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='main-header'>♻️ RECICLAJE INTELIGENTE</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subheader'>Aprende a clasificar correctamente tus residuos</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-box'>El reciclaje adecuado es esencial para proteger nuestro planeta. Esta aplicación te ayudará a determinar en qué caneca debes depositar cada tipo de residuo.</div>", unsafe_allow_html=True)
    
    # Información de las canecas
    st.markdown("### Conoce las canecas de reciclaje")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<div class='caneca-info' style='background-color: {color_map['azul'][1]};'><h3>Caneca Azul</h3><p>Para materiales reciclables como papel, cartón, plástico, vidrio y metal.</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='caneca-info' style='background-color: {color_map['verde'][1]};'><h3>Caneca Verde</h3><p>Para residuos orgánicos como restos de comida y material vegetal.</p></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div class='caneca-info' style='background-color: {color_map['gris'][1]};'><h3>Caneca Gris</h3><p>Para residuos no reciclables como icopor, pañales y envoltorios metalizados.</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='caneca-info' style='background-color: {color_map['roja'][1]};'><h3>Caneca Roja</h3><p>Para residuos peligrosos como pilas, medicamentos y productos químicos.</p></div>", unsafe_allow_html=True)
    
    # Botón para ir al clasificador con acción directa
    if st.button("Ir al Clasificador", type="primary", key="ir_clasificador"):
        st.session_state['pagina'] = "Clasificador"
        st.rerun()

# Página del clasificador
elif pagina == "Clasificador":
    st.markdown("<h1 class='main-header'>Clasificador de Residuos</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Descubre en qué caneca debes depositar tu residuo</p>", unsafe_allow_html=True)
    
    # Campo de entrada para el residuo a clasificar
    residuo_input = st.text_input("¿Qué residuo quieres clasificar?", placeholder="Ej: botella plástica")
    
    # Botón de clasificación
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        boton_clasificar = st.button("Clasificar", type="primary", use_container_width=True)
    
    if boton_clasificar and residuo_input:
        # Extracción de características
        entrada = {
            'es_plastico': int(any(p in residuo_input.lower() for p in 
                ['plást', 'plast', 'icop', 'bolsa', 'pet', 'pvc', 'poliestireno', 'envase', 'detergente', 'botella', 'vaso', 'juguete'])),
            
            'es_organico': int(any(o in residuo_input.lower() for p in 
                ['banana', 'naranja', 'comida', 'casc', 'fruta', 'vegetal', 'pan', 'café', 'huevo', 'servilleta'] for o in residuo_input.lower().split())),
            
            'es_papel': int(any(p in residuo_input.lower() for p in 
                ['papel', 'revista', 'carton', 'cartón', 'periódico', 'cuaderno', 'libro', 'folleto', 'higiénico', 'cereal'])),
            
            'es_vidrio': int(any(v in residuo_input.lower() for v in 
                ['vidrio', 'cristal', 'espejo', 'bombilla'])),
            
            'es_metal': int(any(m in residuo_input.lower() for m in 
                ['lata', 'metal', 'aluminio', 'aerosol', 'clip', 'grapa', 'clavo', 'tapa'])),
            
            'es_peligroso': int(any(p in residuo_input.lower() for p in 
                ['pila', 'batería', 'medicamento', 'jeringa', 'bombilla ahorradora', 'termómetro',
                'pintura', 'aceite', 'pesticida', 'tinta', 'peligroso', 'tóxico', 'químico'])),
            
            'es_higienico': int(any(h in residuo_input.lower() for h in 
                ['pañal', 'toalla higiénica', 'hilo dental', 'cepillo', 'esponja']))
        }
        
        # Predicción
        entrada_df = pd.DataFrame([entrada])
        
        # Verificación de residuos peligrosos
        if entrada['es_peligroso'] == 1:
            pred = 'roja'
        else:
            pred = modelo.predict(entrada_df)[0]
        
        # Mostrar resultado
        mensaje, color, explicacion = color_map[pred]
        
        st.markdown(
            f"""
            <div style='background-color:{color}; padding: 1.5rem; border-radius: 0.5rem; margin: 1.5rem 0;'>
                <h3 style='color:white; margin-bottom: 0.5rem;'>{mensaje}</h3>
                <p style='color:white;'>{explicacion}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Mostrar recomendaciones específicas
        st.subheader("Recomendaciones:")
        
        if pred == 'azul':
            st.info("✅ Asegúrate de limpiar y enjuagar el envase antes de reciclarlo.")
            st.info("✅ Aplasta los envases plásticos y cajas para reducir su volumen.")
        elif pred == 'verde':
            st.info("✅ Los residuos orgánicos pueden ser compostados en casa para generar abono.")
            st.info("✅ Corta los residuos grandes en trozos más pequeños para facilitar su descomposición.")
        elif pred == 'gris':
            st.info("✅ Intenta reducir el consumo de productos no reciclables en tu día a día.")
            st.info("✅ Busca alternativas sostenibles para estos productos.")
        elif pred == 'roja':
            st.warning("⚠️ Nunca mezcles estos residuos con la basura común.")
            st.warning("⚠️ Llévalos a un punto de recolección especializado.")
    
    # Sugerencias de residuos comunes
    with st.expander("👇 Ver ejemplos de residuos comunes"):
        st.write("Aquí hay algunos ejemplos que puedes probar:")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**Reciclables:**")
            st.markdown("- Botella plástica")
            st.markdown("- Periódico")
            st.markdown("- Lata de refresco")
            st.markdown("- Caja de cartón")
        
        with col2:
            st.markdown("**Orgánicos:**")
            st.markdown("- Cáscara de banana")
            st.markdown("- Restos de comida")
            st.markdown("- Cáscara de huevo")
            st.markdown("- Restos de frutas")
        
        with col3:
            st.markdown("**No Reciclables:**")
            st.markdown("- Vaso de icopor")
            st.markdown("- Pañal usado")
            st.markdown("- Chicle usado")
            st.markdown("- Envoltura metalizada")
        
        with col4:
            st.markdown("**Peligrosos:**")
            st.markdown("- Pila/batería")
            st.markdown("- Medicamento vencido")
            st.markdown("- Jeringa usada")
            st.markdown("- Pintura")
            
    # Página de "Aprende más"
elif pagina == "Aprende más":
    st.markdown("<h1 class='main-header'>Guía de Reciclaje</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📘 Reciclables", "🌱 Orgánicos", "⚪ No Reciclables", "🔴 Peligrosos"])
    
    with tab1:
        st.header("Residuos Reciclables")
        st.markdown("""
        Los residuos reciclables son aquellos que pueden ser transformados en nuevos productos.
        
        ### ¿Qué va en la caneca azul?
        
        * **Papel y cartón**: periódicos, revistas, cajas, envases de cartón, papel de oficina
        * **Plástico**: botellas, envases, bolsas (preferiblemente limpias y secas)
        * **Vidrio**: botellas y frascos (sin romper)
        * **Metal**: latas, papel aluminio, tapas metálicas
        
        ### Consejos para reciclar correctamente
        
        1. Enjuaga los envases para eliminar residuos de alimentos
        2. Aplasta las botellas y cajas para ahorrar espacio
        3. Quita tapas y etiquetas cuando sea posible
        4. No incluyas papel o cartón mojado o con restos de comida
        """)
        
    with tab2:
        st.header("Residuos Orgánicos")
        st.markdown("""
        Los residuos orgánicos son biodegradables y pueden convertirse en compost.
        
        ### ¿Qué va en la caneca verde?
        
        * **Restos de alimentos**: frutas, verduras, cáscaras
        * **Restos de jardinería**: hojas, ramas pequeñas, flores marchitas
        * **Otros**: cáscaras de huevo, café molido usado, bolsitas de té
        
        ### Beneficios del compostaje
        
        1. Reduce la cantidad de basura enviada a vertederos
        2. Produce abono natural para plantas y jardines
        3. Disminuye la emisión de gases de efecto invernadero
        4. Ahorra dinero en fertilizantes comerciales
        """)
        
    with tab3:
        st.header("Residuos No Reciclables")
        st.markdown("""
        Son aquellos que no pueden ser reciclados ni compostados con los métodos actuales.
        
        ### ¿Qué va en la caneca gris?
        
        * **Plásticos no reciclables**: icopor, algunos juguetes, cepillos de dientes
        * **Productos de higiene**: pañales, toallas higiénicas, hilo dental
        * **Otros**: chicles usados, colillas de cigarrillo, esponjas
        
        ### Alternativas sostenibles
        
        1. Reemplaza productos desechables por reutilizables
        2. Evita el uso de icopor y plásticos de un solo uso
        3. Busca productos con menos empaque
        4. Considera alternativas biodegradables cuando sea posible
        """)
        
    with tab4:
        st.header("Residuos Peligrosos")
        st.markdown("""
        Son residuos que requieren un manejo especial debido a su potencial peligro para la salud o el medio ambiente.
        
        ### ¿Qué va en la caneca roja?
        
        * **Productos químicos**: pinturas, disolventes, pesticidas
        * **Electrónicos**: pilas, baterías
        * **Medicamentos**: medicamentos vencidos, jeringas
        * **Otros**: bombillas ahorradoras, termómetros de mercurio
        
        ### Manejo adecuado
        
        1. Nunca los mezcles con la basura ordinaria
        2. Llévalos a puntos limpios o de recolección especializados
        3. Guárdalos en sus envases originales cuando sea posible
        4. Mantén estos productos fuera del alcance de niños y mascotas
        """)
    
    # Centros de reciclaje cercanos
    st.subheader("Centros de reciclaje cercanos")
    st.write("Encuentra lugares donde puedes llevar tus residuos para un manejo adecuado:")
    
    st.markdown("""
    <div style="width: 100%; height: 450px; margin-bottom: 20px; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <iframe 
            src="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d254508.51141489705!2d-74.2478979908203!3d4.6482837!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1spuntos%20de%20reciclaje%20bogota!5e0!3m2!1ses!2sco!4v1711478637486!5m2!1ses!2sco" 
            width="100%" 
            height="450" 
            style="border:0;" 
            allowfullscreen="" 
            loading="lazy" 
            referrerpolicy="no-referrer-when-downgrade">
        </iframe>
    </div>    """, unsafe_allow_html=True)
    
# Página de estadísticas
elif pagina == "Estadísticas":
    st.markdown("<h1 class='main-header'>Estadísticas e Impacto</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Conoce los datos sobre reciclaje y su impacto ambiental</p>", unsafe_allow_html=True)
    
    # Distribución de residuos en la base de datos
    st.subheader("Distribución de residuos por tipo")
    
    # Crear gráfica de distribución
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Contar los tipos de residuos
    conteo = residuos['tipo'].value_counts()
    colores = {'reciclable': '#0066CC', 'organico': '#33A532', 'no reciclable': '#555555', 'peligroso': '#B30000'}
    
    # Crear gráfico de barras
    bars = ax.bar(conteo.index, conteo.values, color=[colores[tipo] for tipo in conteo.index])
    
    # Añadir etiquetas
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}', ha='center', va='bottom')
    
    ax.set_ylabel('Cantidad')
    ax.set_title('Distribución de residuos en la base de datos')
    fig.tight_layout()
    
    st.pyplot(fig)
    
    # Estadísticas importantes
    st.subheader("Datos de impacto ambiental")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='stat-box'>
            <h3>1 tonelada</h3>
            <p>de papel reciclado salva 17 árboles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='stat-box'>
            <h3>1 botella plástica</h3>
            <p>puede tardar 450 años en descomponerse</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='stat-box'>
            <h3>29%</h3>
            <p>de los residuos son reciclables pero van a vertederos</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Materiales por tiempo de degradación
    st.subheader("Tiempo de degradación por tipo de material")
    
    datos_degradacion = {
        'Material': ['Papel', 'Cáscara de plátano', 'Botella plástica', 'Lata de aluminio', 'Vidrio', 'Pilas'],
        'Tiempo (años)': [1, 0.2, 450, 200, 4000, 1000]
    }
    
    df_degradacion = pd.DataFrame(datos_degradacion)
    
    # Gráfico horizontal
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Colores según tipo de material
    colores_materiales = ['#0066CC', '#33A532', '#0066CC', '#0066CC', '#0066CC', '#B30000']
    
    # Crear gráfico de barras horizontal
    bars2 = ax2.barh(df_degradacion['Material'], df_degradacion['Tiempo (años)'], color=colores_materiales)
    
    # Escala logarítmica para mejor visualización
    ax2.set_xscale('log')
    
    # Añadir etiquetas
    for bar in bars2:
        width = bar.get_width()
        label_x_pos = width * 1.1
        ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f} años', 
                 va='center', ha='left')
    
    ax2.set_xlabel('Tiempo de degradación (años, escala logarítmica)')
    ax2.set_title('¿Cuánto tarda en degradarse la basura?')
    fig2.tight_layout()
    
    st.pyplot(fig2)
    
    # Consejos para reducir residuos
    st.subheader("Consejos para reducir tus residuos")
    
    with st.expander("Ver consejos prácticos"):
        st.markdown("""
        ### Filosofía de las "5R"
        
        1. **Rechazar** lo que no necesitas
        2. **Reducir** lo que consumes
        3. **Reutilizar** lo que sea posible
        4. **Reparar** antes de desechar
        5. **Reciclar** como última opción
        
        ### Consejos prácticos
        
        - Usa bolsas de tela reutilizables para las compras
        - Lleva tu propia botella de agua reutilizable
        - Compra a granel para reducir empaques
        - Composta tus residuos orgánicos
        - Repara los objetos rotos antes de desecharlos
        - Dona la ropa y objetos que ya no uses
        """)
    
    # Huella ecológica
    st.subheader("Calculadora de impacto")
    st.markdown("Calcula tu impacto ambiental basado en tus hábitos de consumo y reciclaje.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Hábitos de consumo")
        uso_bolsas = st.slider("¿Cuántas bolsas plásticas usas a la semana?", 0, 20, 5)
        uso_botellas = st.slider("¿Cuántas botellas plásticas desechables usas a la semana?", 0, 20, 3)
        compra_granel = st.slider("¿Qué porcentaje de tus compras son a granel?", 0, 100, 20)
    
    with col2:
        st.markdown("#### Hábitos de reciclaje")
        porcentaje_reciclaje = st.slider("¿Qué porcentaje de tus residuos reciclas?", 0, 100, 30)
        composta = st.checkbox("¿Haces compostaje de residuos orgánicos?")
        reutiliza = st.checkbox("¿Reutilizas envases y objetos?")
    
    # Calcular impacto
    if st.button("Calcular mi impacto"):
        # Cálculo muy simplificado (en la vida real sería más complejo)
        puntaje_base = 100  # Impacto base
        
       
        puntaje_base += uso_bolsas * 2  
        puntaje_base += uso_botellas * 3  
        puntaje_base -= compra_granel * 0.5  
        puntaje_base -= porcentaje_reciclaje * 0.8  
        
        if composta:
            puntaje_base -= 20  
        
        if reutiliza:
            puntaje_base -= 15  
        
        puntaje_final = max(0, puntaje_base)
        
        
        if puntaje_final < 50:
            categoria = "Bajo impacto - ¡Excelente!"
            color = "#33A532"
        elif puntaje_final < 100:
            categoria = "Impacto moderado - Puedes mejorar"
            color = "#FFA500"
        else:
            categoria = "Alto impacto - Necesitas cambios"
            color = "#B30000"
        
        st.markdown(f"""
        <div style='background-color:{color}; padding:1.5rem; border-radius:0.5rem; color:white; text-align:center; margin-top:1rem;'>
            <h2>Tu impacto ambiental: {puntaje_final:.0f} puntos</h2>
            <h3>{categoria}</h3>
        </div>
        """, unsafe_allow_html=True)
        
    
        st.subheader("Recomendaciones personalizadas")
        
        if uso_bolsas > 5:
            st.info("📝 Considera usar bolsas de tela reutilizables para reducir tu consumo de bolsas plásticas.")
        
        if uso_botellas > 3:
            st.info("📝 Invierte en una botella reutilizable de buena calidad para reducir drásticamente tu impacto.")
        
        if compra_granel < 30:
            st.info("📝 Busca tiendas locales que ofrezcan productos a granel para reducir empaques.")
        
        if porcentaje_reciclaje < 50:
            st.info("📝 Mejora tu sistema de clasificación de residuos en casa para aumentar tu porcentaje de reciclaje.")
        
        if not composta:
            st.info("📝 Considera iniciar un compostaje casero para tus residuos orgánicos, es más sencillo de lo que crees.")
st.markdown("<div class='footer'>© 2025 - Aplicación de Reciclaje Inteligente - Desarrollado con ♥ para un planeta más limpio</div>", unsafe_allow_html=True)